"""Streamlit front end.

    streamlit run app.py

This file is presentation only. Every import from legal_ai is a call into logic
that runs identically from the CLI, and nothing in src/ imports Streamlit. That
separation is deliberate: it keeps the pipeline testable without a browser, and
it means putting a FastAPI layer or a different front end on top later is a
matter of calling the same functions, not untangling them.

The disclaimer is rendered on every screen rather than once at the start. A user
who lands mid-report should not have to scroll to learn the tool is not a lawyer.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

# Streamlit Cloud supplies secrets through st.secrets, not the environment, so a
# key configured correctly in the dashboard would sit there unread and the app
# would fail at startup claiming no key was set.
#
# Copied into os.environ HERE, before legal_ai is imported, for two reasons.
# First, config.py reads LLM_PROVIDER at import time, so a later copy would be
# too late. Second, this keeps the Streamlit dependency in the front end: config
# .py never learns that Streamlit exists, and the pipeline stays runnable from
# the CLI and from tests with no browser and no Streamlit install.
#
# Existing environment variables win, so a local .env still takes precedence.
for _secret in ("LLM_PROVIDER", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
    if _secret not in os.environ:
        try:
            if _secret in st.secrets:
                os.environ[_secret] = str(st.secrets[_secret])
        except Exception:
            # No secrets.toml at all is the normal local case, and st.secrets
            # raises rather than returning empty. Not an error worth surfacing.
            pass

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from legal_ai.corpus import load_stats  # noqa: E402
from legal_ai.extract import extract_document  # noqa: E402
from legal_ai.negotiate import draft_email, negotiate  # noqa: E402
from legal_ai.profile import build_profile  # noqa: E402
from legal_ai.scoring import Severity, score_document  # noqa: E402

st.set_page_config(page_title="NDA Reviewer", page_icon="⚖️", layout="wide")

# Badge background paired with WHITE text, so each colour must clear WCAG 2.1 AA
# (4.5:1) against #ffffff -- the badge text is 12px, which is below the 18.66px
# threshold for the relaxed 3:1 "large text" rule, so the full ratio applies.
#
# Severity is never carried by colour alone: every badge states its level in
# words, because ~4% of men cannot reliably separate the red one from the amber
# one, and a risk indicator that only some readers can read is not an indicator.
# tests/test_accessibility.py recomputes these ratios so the claim stays true
# if someone retunes the palette.
SEVERITY_STYLE = {
    Severity.HIGH: ("#b3261e", "HIGH RISK"),
    # Was #b26b00, which measured 4.20:1 and failed AA. Darkened to 5.43:1.
    Severity.MEDIUM: ("#9a5b00", "MODERATE"),
    Severity.LOW: ("#5f6368", "MINOR"),
    Severity.FAVOURABLE: ("#1e7b34", "IN YOUR FAVOUR"),
}

DISCLAIMER = (
    "This tool provides information, not legal advice, and does not replace a "
    "lawyer. It compares your document against a small reference corpus."
)


@contextmanager
def _saved_upload(uploaded):
    """Persist an upload to a temp file for the duration of the `with` block.

    Suffix is preserved: the parser dispatches on it, and a .docx written
    without its extension would be read as plain text and produce nonsense.

    A context manager rather than a plain function because the file MUST be
    removed again. An upload here is someone's draft legal agreement, and
    NamedTemporaryFile(delete=False) leaves it on disk for the life of the
    process. On the public deployment that means every document anyone uploads
    accumulating in the server's temp directory, one visitor's NDA outliving
    their session and sitting beside the next visitor's. The pipeline only needs
    the path long enough to read the text, so the window is closed to that.
    """
    suffix = Path(uploaded.name).suffix or ".txt"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        handle.write(uploaded.getbuffer())
        handle.close()
        yield Path(handle.name)
    finally:
        # missing_ok: never let cleanup failure mask the real error above it.
        Path(handle.name).unlink(missing_ok=True)


def _severity_badge(severity: Severity) -> str:
    colour, label = SEVERITY_STYLE[severity]
    return (
        f"<span style='background:{colour};color:white;padding:2px 8px;"
        f"border-radius:3px;font-size:0.75rem;font-weight:600'>{label}</span>"
    )


# --- Sidebar ----------------------------------------------------------------

with st.sidebar:
    st.header("About")
    st.caption(DISCLAIMER)

    try:
        stats = load_stats()
        st.metric("Reference NDAs", stats.n_documents)
        if not stats.is_credible:
            st.warning(stats.credibility_note(), icon="⚠️")
        st.caption(f"Corpus: {stats.provenance}")
        st.caption(f"Built {stats.built_on}")
    except RuntimeError:
        stats = None
        st.error("No corpus statistics found. Run `py build_corpus.py` first.")

    st.divider()
    st.caption(
        "Analysis is cached by document content, so re-running the same file "
        "costs nothing and returns instantly."
    )


# --- Main -------------------------------------------------------------------

st.title("NDA Reviewer")
st.caption(
    "Upload an NDA to see which terms differ from market-standard agreements, "
    "why that matters, and what to ask for instead."
)

if stats is None:
    st.stop()

uploaded = st.file_uploader(
    "Upload an NDA", type=["pdf", "docx", "txt", "md"],
    help="Text-layer PDFs only. Scanned documents are not supported.",
)

if uploaded is None:
    st.info(
        "No document yet. You can try one of the sample NDAs in `data/golden/` "
        "— `aggressive_nda_01.txt` has eight deliberately unusual terms."
    )
    st.stop()

try:
    with _saved_upload(uploaded) as path, st.spinner(
        "Reading the document and extracting terms..."
    ):
        extraction = extract_document(path)
except (RuntimeError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

profile = build_profile(Path(uploaded.name).stem, extraction.extracted)
report = score_document(profile, stats)

# --- Summary ----------------------------------------------------------------

# st.header, not st.subheader: these are the two top-level sections under the
# page title, and subheader renders <h3>, which would skip a level and leave a
# screen reader's heading outline reading h1 -> h3 with nothing in between.
st.header(report.headline())

high = report.count(Severity.HIGH)
medium = report.count(Severity.MEDIUM)
low = report.count(Severity.LOW)

columns = st.columns(4)
columns[0].metric("High risk", high)
columns[1].metric("Moderate", medium)
columns[2].metric("Minor", low)
columns[3].metric("Clauses read", len(extraction.extracted))

st.caption(report.corpus_note)

if not report.deviations:
    st.success("Nothing in this document deviates materially from the reference corpus.")

# --- Findings ---------------------------------------------------------------

for deviation in report.deviations:
    heading = None
    if deviation.clause_index is not None and deviation.clause_index < len(extraction.raw_clauses):
        heading = extraction.raw_clauses[deviation.clause_index].heading

    with st.container(border=True):
        left, right = st.columns([3, 1])
        left.markdown(f"**{deviation.label}**" + (f" — {heading}" if heading else ""))
        right.markdown(_severity_badge(deviation.severity), unsafe_allow_html=True)

        st.markdown(f"**This document:** {deviation.finding}")
        st.markdown(f"**Reference NDAs:** {deviation.market}")
        st.markdown(f"**Why it matters:** {deviation.why_it_matters}")

        # The clause text is collapsed rather than shown inline: users want the
        # finding first, and the source only when they doubt it.
        if deviation.clause_index is not None and deviation.clause_index < len(extraction.raw_clauses):
            with st.expander("Show the clause this came from"):
                st.text(extraction.raw_clauses[deviation.clause_index].text)

if report.skipped:
    st.caption(
        "Not checked, because the reference corpus has no baseline for them: "
        + ", ".join(sorted(set(report.skipped)))
    )

# --- Negotiation ------------------------------------------------------------

st.divider()

if not report.risks:
    st.stop()

st.header("Negotiating positions")
st.caption(
    "Argues both sides of each flagged term and suggests replacement wording. "
    "This makes one additional API call."
)

if st.button("Generate negotiating positions", type="primary"):
    try:
        with st.spinner("Arguing both sides of each flagged term..."):
            st.session_state.negotiations = negotiate(report, extraction.raw_clauses)
    except RuntimeError as exc:
        st.error(str(exc))

negotiations = st.session_state.get("negotiations")

if negotiations:
    st.info(negotiations.overall_assessment)

    for item in negotiations.negotiations:
        with st.expander(f"{item.risk_level.value.upper()} — {item.attribute}"):
            st.markdown(f"**What it means:** {item.plain_explanation}")
            st.markdown(f"**Their case:** {item.counterparty_justification}")
            st.markdown(f"**Your case:** {item.your_position}")
            st.markdown("**Ask for this wording:**")
            st.code(item.suggested_redline, language=None)
            st.markdown(f"**Fall back to:** {item.fallback_position}")
            st.success(f"Say this: \"{item.talking_point}\"")

    gaps = negotiations.disagreements(report)
    if gaps:
        with st.expander("Where statistical rarity and practical risk differ"):
            st.caption(
                "The scorer measures how unusual a term is. The negotiation stage "
                "judges what it would actually do to you. These are different "
                "questions, and the gap is worth seeing."
            )
            for attribute, statistical, practical in gaps:
                st.markdown(
                    f"- **{attribute}** — unusual: `{statistical}`, "
                    f"practical risk: `{practical}`"
                )

    st.divider()
    if st.button("Draft a negotiation email"):
        try:
            with st.spinner("Drafting..."):
                st.session_state.email = draft_email(negotiations)
        except RuntimeError as exc:
            st.error(str(exc))

    if st.session_state.get("email"):
        st.text_area("Draft email", st.session_state.email, height=320)
        st.download_button(
            "Download as .txt",
            st.session_state.email,
            file_name=f"negotiation-{Path(uploaded.name).stem}.txt",
        )

st.divider()
st.caption(DISCLAIMER)
