"""Turn flagged deviations into negotiating positions. The generative stage.

WHERE THE GENERATIVE AI ACTUALLY LIVES

Extraction turns prose into typed facts; scoring is arithmetic. This module is
the part that writes something new -- it argues both sides of each flagged term
and produces language a user can actually send. If asked "where is the GenAI in
this project?", the honest answer is: extraction does the reading, this does the
writing, and the statistics in between exist to keep the writing grounded.

ONE CALL, TWO STANCES -- NOT A MULTI-TURN DEBATE

The plan called for two persona agents debating and a third synthesising. That
is three sequential calls per flagged clause: with eight flagged clauses, 24
calls and 30-60 seconds of latency, during a live demo. It also tends to produce
mush, because each round regresses toward the mean of the two positions.

Instead the model produces both stances and the synthesis in a single structured
response. The schema forces it to state the counterparty's justification before
arguing against it, which is what actually makes the output useful -- a
negotiating position that cannot articulate the other side's reasoning is not a
negotiating position, it is a complaint. Same analytical content, one call.

GROUNDING

Every negotiation is handed the specific corpus comparison that triggered the
flag. The prompt requires the argument to cite it. That is the difference
between "this term seems aggressive" (an opinion the user cannot check) and
"this runs 36 months where every reference NDA uses 12" (a position they can
take into a conversation).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from legal_ai import config
from legal_ai.llm import LLMError, structured, text
from legal_ai.parsing import RawClause
from legal_ai.scoring import DeviationReport, Severity

# Bump when the schemas below change shape, so cached results from an older
# schema are not served as current.
NEGOTIATION_SCHEMA_VERSION = "1"


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        """Higher is more urgent, matching scoring.Severity.rank.

        Kept as a property rather than an inline dict at the one call site,
        so that this ordering and Severity's are recognisably the same idea
        written the same way -- they are separate enums on purpose (statistical
        rarity is not practical risk), but nothing is served by them being
        ordered by two differently-shaped literals.
        """
        return {"high": 3, "medium": 2, "low": 1}[self.value]


class ClauseNegotiation(BaseModel):
    """One flagged term, argued from both sides and resolved into an ask."""

    attribute: str = Field(
        description="The attribute identifier given in the input, copied exactly."
    )
    risk_level: RiskLevel = Field(
        description="Practical risk to the person signing. Judge the real-world "
        "consequence, not merely how statistically unusual the term is. A term can "
        "be uncommon yet harmless, or common yet serious."
    )
    plain_explanation: str = Field(
        description="Two sentences maximum, for someone with no legal training. "
        "Say what the clause does to them in practice. No jargon, no hedging, no "
        "restating the clause in legal language."
    )
    counterparty_justification: str = Field(
        description="The strongest honest reason the other side would want this "
        "term. Argue it properly -- steelman it. If the term is genuinely "
        "indefensible, say that plainly rather than inventing a weak excuse."
    )
    your_position: str = Field(
        description="The argument for changing it, citing the specific market "
        "comparison provided. Address the counterparty's justification directly "
        "rather than ignoring it."
    )
    suggested_redline: str = Field(
        description="Concrete replacement wording, ready to paste into the "
        "document. Actual contract language, not a description of what to change."
    )
    fallback_position: str = Field(
        description="What to accept if they refuse the redline. A negotiation "
        "with no fallback is an ultimatum. If the term is one that should not be "
        "conceded at all, say so and explain why."
    )
    talking_point: str = Field(
        description="One sentence the user can say out loud in a conversation or "
        "email. Direct, non-confrontational, citing the market comparison."
    )


class NegotiationSet(BaseModel):
    """Top-level response covering every flagged term in one document."""

    negotiations: list[ClauseNegotiation]
    overall_assessment: str = Field(
        description="Two or three sentences on the document as a whole: is this a "
        "normal agreement with a few rough edges, or is it one-sided enough that "
        "the user should think carefully before signing? Be direct."
    )


@dataclass
class NegotiationResult:
    negotiations: list[ClauseNegotiation]
    overall_assessment: str
    from_cache: bool

    def by_attribute(self) -> dict[str, ClauseNegotiation]:
        return {n.attribute: n for n in self.negotiations}

    def disagreements(self, report: DeviationReport) -> list[tuple[str, str, str]]:
        """Where the model's practical risk rating differs from the statistical
        severity. Genuinely informative rather than a bug: "unusual" and "risky"
        are different questions, and a gap between them is worth showing.
        Returns (attribute, statistical_severity, practical_risk)."""
        by_attribute = self.by_attribute()
        gaps = []
        for deviation in report.deviations:
            negotiation = by_attribute.get(deviation.attribute)
            if negotiation is None or deviation.severity is Severity.FAVOURABLE:
                continue
            if negotiation.risk_level.value != deviation.severity.value:
                gaps.append(
                    (deviation.attribute, deviation.severity.value,
                     negotiation.risk_level.value)
                )
        return gaps


SYSTEM_PROMPT = """You are a contracts negotiator helping someone who is not a \
lawyer decide what to push back on in an NDA they have been asked to sign.

You are given terms that deviate from a reference corpus of market-standard NDAs, \
each with the specific comparison that flagged it.

Rules:

1. Argue both sides. State the counterparty's strongest honest reason for wanting \
the term before arguing against it. A position that cannot articulate the other \
side's reasoning is a complaint, not a negotiating position. Where a term is \
genuinely indefensible, say so rather than manufacturing a weak justification.

2. Cite the comparison you were given. "This runs 36 months where every reference \
NDA uses 12" is a position someone can take into a conversation. "This seems \
aggressive" is not.

3. Write redlines as actual contract language, ready to paste. Not a description \
of the change -- the replacement text itself.

4. Always give a fallback. A negotiation with no fallback is an ultimatum, and \
most of these terms are negotiable in degree rather than in principle. Where a \
term genuinely should not be conceded, say that explicitly and explain why.

5. Rate practical risk, not statistical rarity. You are told how unusual each term \
is; judge separately what it would actually do to this person. Uncommon terms can \
be harmless and common ones can be serious.

6. Write for a non-lawyer throughout. Short sentences. No Latin, no "heretofore", \
no hedging. The user should be able to read your talking point aloud without \
rehearsing it.

7. You provide information, not legal advice. Do not tell the user whether to \
sign. Give them what they need to decide and to negotiate."""


def _cache_path(payload: str) -> Path:
    digest = hashlib.sha256(
        f"{payload}|{config.NEGOTIATION_MODEL}|{NEGOTIATION_SCHEMA_VERSION}".encode()
    ).hexdigest()[:16]
    return config.CACHE_DIR / f"negotiate_{digest}.json"


def _format_deviations(report: DeviationReport, clauses: list[RawClause]) -> str:
    """Render flagged terms with their corpus comparison and the clause text.

    The clause text is included because a redline has to fit the document's own
    wording; a suggested replacement written without seeing the original reads
    like it came from a different agreement.
    """
    blocks = []
    for deviation in report.risks:
        lines = [
            f"--- TERM: {deviation.attribute} ---",
            f"Label: {deviation.label}",
            f"Statistical severity: {deviation.severity.value}",
            f"This document: {deviation.finding}",
            f"Reference corpus: {deviation.market}",
            f"Why it matters: {deviation.why_it_matters}",
        ]
        if deviation.clause_index is not None and deviation.clause_index < len(clauses):
            clause = clauses[deviation.clause_index]
            heading = clause.heading or f"clause {clause.index + 1}"
            lines.append(f"Clause text ({heading}):\n{clause.text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def negotiate(
    report: DeviationReport,
    clauses: list[RawClause],
    *,
    use_cache: bool = True,
) -> NegotiationResult:
    """Produce negotiating positions for every flagged term in one call.

    Returns an empty result when nothing was flagged -- a clean document needs no
    negotiation, and calling the API to say so would waste a request.
    """
    if not report.risks:
        return NegotiationResult(
            negotiations=[],
            overall_assessment=(
                "Nothing in this document deviates materially from the reference "
                "corpus. There is no specific term flagged for negotiation."
            ),
            from_cache=False,
        )

    payload = _format_deviations(report, clauses)
    cache_file = _cache_path(payload)

    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return NegotiationResult(
            negotiations=[
                ClauseNegotiation.model_validate(n) for n in cached["negotiations"]
            ],
            overall_assessment=cached["overall_assessment"],
            from_cache=True,
        )

    user_content = (
        f"The following {len(report.risks)} terms in this NDA deviate from the "
        f"reference corpus. {report.corpus_note}\n\n"
        f"Produce a negotiating position for each.\n\n{payload}"
    )

    try:
        parsed = structured(
            SYSTEM_PROMPT, user_content, NegotiationSet,
            model=config.NEGOTIATION_MODEL,
        )
    except LLMError as exc:
        raise RuntimeError(
            f"Negotiation failed: {exc} The deviation report is unaffected."
        ) from exc

    cache_file.write_text(
        json.dumps(
            {
                "negotiations": [n.model_dump(mode="json") for n in parsed.negotiations],
                "overall_assessment": parsed.overall_assessment,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return NegotiationResult(
        negotiations=list(parsed.negotiations),
        overall_assessment=parsed.overall_assessment,
        from_cache=False,
    )


# --- Email draft -------------------------------------------------------------

EMAIL_SYSTEM_PROMPT = """You draft short, professional negotiation emails for \
someone asking for changes to an NDA before signing.

Rules:

1. Cooperative in tone, specific in content. The reader should feel this is a \
routine request from someone who intends to sign, not an objection.

2. Ask for the highest-priority changes only -- at most four. An email listing \
every deviation reads as obstruction and gets negotiated as a block instead of \
point by point.

3. Cite the market comparison for each ask, briefly. It converts the request from \
a preference into a norm.

4. Short. Under 250 words. Numbered asks, each one or two sentences.

5. No legal threats, no Latin, no invented facts about the sender's circumstances.

Output only the email body. No subject line, no placeholders like [YOUR NAME]."""


def draft_email(result: NegotiationResult, *, use_cache: bool = True) -> str:
    """Draft a negotiation email from the highest-priority positions.

    Separate call rather than part of the negotiation response, because it is a
    user-initiated action in the UI -- most users read the report without ever
    sending an email, and generating one every time would be wasted spend.
    """
    if not result.negotiations:
        return "No terms were flagged for negotiation, so there is nothing to ask for."

    priority = sorted(
        result.negotiations,
        key=lambda n: -n.risk_level.rank,
    )[:4]

    payload = "\n\n".join(
        f"ASK: {n.attribute}\n"
        f"Risk: {n.risk_level.value}\n"
        f"Position: {n.your_position}\n"
        f"Requested change: {n.suggested_redline}\n"
        f"Fallback: {n.fallback_position}"
        for n in priority
    )

    cache_file = _cache_path(f"email|{payload}")
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))["email"]

    try:
        email = text(
            EMAIL_SYSTEM_PROMPT,
            f"Draft the email requesting these changes:\n\n{payload}",
            model=config.NEGOTIATION_MODEL,
            max_tokens=2000,
        )
    except LLMError as exc:
        raise RuntimeError(f"Could not draft the email: {exc}") from exc

    cache_file.write_text(json.dumps({"email": email}), encoding="utf-8")
    return email
