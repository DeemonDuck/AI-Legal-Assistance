"""Build the market-standard distributions the deviation scorer compares against.

    py build_corpus.py              build from data/corpus/, save, print summary
    py build_corpus.py --show       print the saved stats without rebuilding
    py build_corpus.py --no-cache   force fresh extraction of every document

Makes one API call per uncached corpus document. With the 8 seed templates that
is 8 calls on the first run and zero thereafter, because extractions are cached
by content hash.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from legal_ai.corpus import (  # noqa: E402
    MIN_DOCUMENTS_FOR_CREDIBLE_STATS,
    build_corpus,
    load_stats,
    save_stats,
)


def _print_summary(stats) -> None:
    print("\n" + "=" * 74)
    print(f"CORPUS STATISTICS  --  {stats.n_documents} documents, built {stats.built_on}")
    print(f"Provenance: {stats.provenance}")
    version = (
        f"v{stats.schema_version}" if stats.schema_version
        else "unrecorded (built before versions were stamped)"
    )
    print(f"Extraction schema: {version}")
    print("=" * 74)

    if stats.numeric:
        print("\nDURATIONS (months) -- percentiles over finite values only")
        header = (
            f"  {'attribute':<24} {'n':>3} {'min':>6} {'p25':>6} "
            f"{'med':>6} {'p75':>6} {'max':>6}  perpetual"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, s in stats.numeric.items():
            if s.median is None:
                print(f"  {name:<24} {s.n:>3}  {'(all perpetual)':>32}  {s.n_perpetual}")
                continue
            print(
                f"  {name:<24} {s.n:>3} {s.minimum:>6.0f} {s.p25:>6.1f} "
                f"{s.median:>6.1f} {s.p75:>6.1f} {s.maximum:>6.0f}  {s.n_perpetual}"
            )

    if stats.boolean:
        print("\nSTRUCTURAL FEATURES")
        for name, s in stats.boolean.items():
            pct = s.true_fraction * 100
            bar = "#" * round(pct / 5)
            print(f"  {name:<30} {s.n_true:>2}/{s.n:<2} {pct:5.1f}%  {bar}")

    if stats.carve_outs.n:
        print(f"\nSTANDARD CARVE-OUTS (present in how many of {stats.carve_outs.n} documents)")
        for term, count in sorted(stats.carve_outs.counts.items(), key=lambda kv: -kv[1]):
            pct = stats.carve_outs.fraction(term) * 100
            print(f"  {term:<42} {count:>2}  {pct:5.1f}%")

    if stats.jurisdictions.n:
        print("\nGOVERNING LAW")
        for name, count in sorted(stats.jurisdictions.counts.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<30} {count}")

    if stats.dispute_forums.n:
        print("\nDISPUTE FORUM")
        for name, count in sorted(stats.dispute_forums.counts.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<30} {count}")

    print("\n" + "-" * 74)
    print(stats.credibility_note())
    if not stats.is_credible:
        print(
            f"Add more templates to data/corpus/ to reach "
            f"{MIN_DOCUMENTS_FOR_CREDIBLE_STATS}+ and rerun."
        )
    print("-" * 74)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build market-standard NDA statistics.")
    parser.add_argument("--show", action="store_true", help="Print saved stats, don't rebuild.")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh extraction.")
    parser.add_argument(
        "--provenance",
        default="synthetic reference drafts shipped with the repository",
        help="Describe where these documents came from. Recorded in the output and "
        "shown to users, so keep it accurate.",
    )
    args = parser.parse_args()

    try:
        if args.show:
            stats = load_stats()
        else:
            print("Building corpus...")
            stats = build_corpus(use_cache=not args.no_cache, provenance=args.provenance)
            path = save_stats(stats)
            print(f"\nSaved to {path}")
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    _print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
