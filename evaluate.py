"""Measure the deviation scorer against hand-written ground truth.

    py evaluate.py              run the eval, print the scorecard
    py evaluate.py --verbose    also list every finding per document
    py evaluate.py --no-cache   force fresh extraction of the eval documents

Exits non-zero if any document fails, so this can gate a commit.

Requires corpus statistics (py build_corpus.py) and makes one API call per
uncached eval document -- 3 on the first run, zero after.

The number this produces is the honest answer to "how do you know it works?",
and it belongs in the README and the demo. Quote it with the corpus caveat
attached: it measures the scorer against the reference corpus, which is not the
same as measuring it against the real market.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from legal_ai import config  # noqa: E402
from legal_ai.corpus import load_stats  # noqa: E402
from legal_ai.evaluation import (  # noqa: E402
    EvalResult,
    check_leakage,
    evaluate_report,
    load_expectations,
)
from legal_ai.extract import SCHEMA_VERSION, extract_document  # noqa: E402
from legal_ai.profile import build_profile  # noqa: E402
from legal_ai.scoring import score_document  # noqa: E402


def _find_document(name: str) -> Path | None:
    for suffix in (".txt", ".md", ".docx", ".pdf"):
        candidate = config.GOLDEN_DIR / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the deviation scorer.")
    parser.add_argument("--verbose", action="store_true", help="List every finding.")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh extraction.")
    parser.add_argument(
        "--allow-stale-corpus",
        action="store_true",
        help="Run even if the baseline was built under a different extraction "
        "schema. The resulting score is not comparable; use only to reproduce "
        "an earlier measurement deliberately.",
    )
    args = parser.parse_args()

    # A document in both sets silently inflates every score, so this is checked
    # before anything is measured rather than trusted.
    leaked = check_leakage()
    if leaked:
        print(
            "ABORTED: these documents are in BOTH data/corpus/ and data/golden/:\n"
            f"    {', '.join(leaked)}\n"
            "An eval document that is part of its own baseline pulls the median "
            "toward itself and scores as more normal than it is. Remove it from "
            "one of the two directories.",
            file=sys.stderr,
        )
        return 1

    try:
        stats = load_stats()
        expectations = load_expectations()
    except RuntimeError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    # Checked for the same reason as leakage above: it produces a plausible
    # number rather than an error, so nothing downstream can catch it. An eval
    # score is a claim about the whole pipeline, and it is only a claim worth
    # making when both sides of the comparison were read the same way.
    mismatch = stats.schema_mismatch(SCHEMA_VERSION)
    if mismatch:
        if not args.allow_stale_corpus:
            print(f"ABORTED: {mismatch}", file=sys.stderr)
            return 1
        print(f"WARNING: {mismatch}\n(continuing: --allow-stale-corpus)\n")

    result = EvalResult()
    print(f"Evaluating {len(expectations)} documents against "
          f"{stats.n_documents} corpus NDAs...\n")

    for name, expectation in expectations.items():
        path = _find_document(name)
        if path is None:
            print(f"  SKIP  {name}: no matching file in {config.GOLDEN_DIR}")
            continue

        try:
            extraction = extract_document(path, use_cache=not args.no_cache)
        except RuntimeError as exc:
            print(f"  ERROR {name}: {exc}", file=sys.stderr)
            return 1

        profile = build_profile(name, extraction.extracted)
        report = score_document(profile, stats)
        document_result = evaluate_report(report, expectation)
        result.documents.append(document_result)

        mark = "PASS" if document_result.passed else "FAIL"
        print(f"  {mark}  {name}")
        print(f"        {document_result.description}")
        print(
            f"        found {len(document_result.found)}/{document_result.n_expected} "
            f"planted terms, {document_result.n_high} HIGH findings"
            + (f" (budget {document_result.max_high})"
               if document_result.max_high is not None else "")
        )

        for missed in document_result.missed:
            print(f"        MISSED  {missed.attribute} -- {missed.notes}")
        for expected, actual in document_result.under_severity:
            print(
                f"        WEAK    {expected.attribute}: got {actual.value}, "
                f"expected at least {expected.min_severity}"
            )
        if document_result.unexpected_high:
            print(
                "        EXTRA HIGH  "
                + ", ".join(document_result.unexpected_high)
            )

        if args.verbose:
            for d in report.deviations:
                print(f"          [{d.severity.value:<10}] {d.attribute}: {d.finding}")
        print()

    print("=" * 74)
    print(result.headline())
    print("=" * 74)
    print(f"\n{stats.credibility_note()}")

    if result.passed:
        print("\nAll documents passed.")
        return 0

    print("\nSome documents failed. See MISSED / WEAK / EXTRA HIGH above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
