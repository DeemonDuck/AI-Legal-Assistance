"""Command-line entry point for the extraction pipeline.

    py analyze.py data/golden/aggressive_nda_01.txt
    py analyze.py --parse-only data/golden/aggressive_nda_01.txt
    py analyze.py --no-cache data/my_nda.pdf

--parse-only runs the parser without touching the API. Use it to check clause
boundaries on a new document before spending tokens on it -- bad boundaries are
the most common cause of bad extraction, and this catches them for free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src/` importable without requiring an editable install. Keeps a fresh
# clone runnable with nothing but `pip install -r requirements.txt`.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from legal_ai.corpus import load_stats  # noqa: E402
from legal_ai.extract import extract_clauses  # noqa: E402
from legal_ai.negotiate import draft_email, negotiate  # noqa: E402
from legal_ai.parsing import parse_document  # noqa: E402
from legal_ai.profile import build_profile  # noqa: E402
from legal_ai.scoring import Severity, score_document  # noqa: E402

SEVERITY_MARK = {
    Severity.HIGH: "[HIGH]",
    Severity.MEDIUM: "[MED ]",
    Severity.LOW: "[LOW ]",
    Severity.FAVOURABLE: "[GOOD]",
}


def _print_parse(clauses) -> None:
    print(f"\nParsed {len(clauses)} clauses\n" + "=" * 78)
    for clause in clauses:
        heading = clause.heading or "(no heading)"
        print(f"\n[{clause.index + 1}] {heading}")
        body = clause.text.replace("\n", " ")
        print(f"    {body[:150]}{'...' if len(body) > 150 else ''}")


def _print_extraction(result) -> None:
    source = "cache" if result.from_cache else "API"
    print(f"\nExtracted {len(result.extracted)} clauses (from {source})\n" + "=" * 78)

    for raw, ext in result.pairs():
        print(f"\n[{raw.index + 1}] {raw.heading or '(no heading)'}")
        print(f"    type      : {ext.clause_type.value}")
        print(f"    favors    : {ext.party_favored.value}")
        print(f"    summary   : {ext.plain_summary}")

        # Only show attributes the model actually populated. Printing the full
        # 16 fields per clause, mostly null, makes the output unreadable.
        populated = {
            name: value
            for name, value in ext.attributes.model_dump().items()
            if value not in (None, [], "")
        }
        if populated:
            print("    attributes:")
            for name, value in populated.items():
                note = " (perpetual)" if value == -1 else ""
                print(f"        {name} = {value}{note}")

        if ext.aggressiveness_signals:
            print(f"    SIGNALS   : {', '.join(ext.aggressiveness_signals)}")


def _print_report(report, clauses) -> None:
    print("\n" + "=" * 78)
    print(f"DEVIATION REPORT  --  {report.document}")
    print("=" * 78)
    print(f"\n{report.headline()}")
    print(f"{report.corpus_note}")

    if not report.deviations:
        print("\nNothing deviates from the reference corpus.")
    else:
        print()

    for d in report.deviations:
        where = ""
        if d.clause_index is not None and d.clause_index < len(clauses):
            heading = clauses[d.clause_index].heading
            if heading:
                where = f"  (see: {heading})"

        print(f"{SEVERITY_MARK[d.severity]} {d.label}{where}")
        print(f"        this document : {d.finding}")
        print(f"        reference NDAs: {d.market}")
        print(f"        why it matters: {d.why_it_matters}")
        print()

    if report.skipped:
        print(
            "Not checked (no baseline in the corpus for these): "
            + ", ".join(sorted(set(report.skipped)))
        )

    print("-" * 78)
    print("This is information, not legal advice, and does not replace a lawyer.")
    print("-" * 78)


def _print_negotiations(result, report) -> None:
    print("\n" + "=" * 78)
    source = "cache" if result.from_cache else "API"
    print(f"NEGOTIATING POSITIONS  ({len(result.negotiations)} terms, from {source})")
    print("=" * 78)
    print(f"\n{result.overall_assessment}\n")

    for n in result.negotiations:
        print("-" * 78)
        print(f"[{n.risk_level.value.upper()}] {n.attribute}")
        print(f"\n  What it means : {n.plain_explanation}")
        print(f"\n  Their case    : {n.counterparty_justification}")
        print(f"\n  Your case     : {n.your_position}")
        print(f"\n  Ask for       : {n.suggested_redline}")
        print(f"\n  Fall back to  : {n.fallback_position}")
        print(f"\n  Say this      : \"{n.talking_point}\"")
        print()

    # Where statistical rarity and practical risk disagree, that gap is itself
    # informative -- "unusual" and "dangerous" are different questions.
    gaps = result.disagreements(report)
    if gaps:
        print("-" * 78)
        print("Where statistical rarity and practical risk differ:")
        for attribute, statistical, practical in gaps:
            print(f"  {attribute}: unusual={statistical}, practical risk={practical}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract structured data from an NDA.")
    parser.add_argument("document", type=Path, help="Path to a PDF, DOCX, or TXT NDA")
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Split into clauses and stop. Makes no API calls and costs nothing.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force a fresh API call instead of reusing a cached extraction.",
    )
    parser.add_argument(
        "--negotiate",
        action="store_true",
        help="Also generate negotiating positions for each flagged term (1 API call).",
    )
    parser.add_argument(
        "--email",
        action="store_true",
        help="Also draft a negotiation email. Implies --negotiate.",
    )
    parser.add_argument(
        "--clauses",
        action="store_true",
        help="Also print the per-clause extraction, not just the deviation report.",
    )
    args = parser.parse_args()

    try:
        clauses = parse_document(args.document)
    except ValueError as exc:
        print(f"Could not read document: {exc}", file=sys.stderr)
        return 1

    if not clauses:
        print("No clauses found -- the document appears to be empty.", file=sys.stderr)
        return 1

    if args.parse_only:
        _print_parse(clauses)
        return 0

    try:
        result = extract_clauses(clauses, use_cache=not args.no_cache)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    if args.clauses:
        _print_extraction(result)

    # Scoring needs a corpus baseline. Missing stats is a setup problem with a
    # one-command fix, so say that rather than failing with a stack trace.
    try:
        stats = load_stats()
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        if not args.clauses:
            print("Extraction succeeded; rerun with --clauses to see it.", file=sys.stderr)
        return 1

    profile = build_profile(args.document.stem, result.extracted)
    report = score_document(profile, stats)
    _print_report(report, result.raw_clauses)

    if not (args.negotiate or args.email):
        return 0

    try:
        negotiations = negotiate(report, result.raw_clauses, use_cache=not args.no_cache)
    except RuntimeError as exc:
        # The deviation report already printed and is still useful on its own,
        # so a negotiation failure must not discard it.
        print(f"\n{exc}", file=sys.stderr)
        return 1

    _print_negotiations(negotiations, report)

    if args.email:
        try:
            print("\n" + "=" * 78)
            print("DRAFT NEGOTIATION EMAIL")
            print("=" * 78 + "\n")
            print(draft_email(negotiations, use_cache=not args.no_cache))
        except RuntimeError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
