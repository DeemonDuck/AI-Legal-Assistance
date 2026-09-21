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

from legal_ai.extract import extract_clauses  # noqa: E402
from legal_ai.parsing import parse_document  # noqa: E402


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

    _print_extraction(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
