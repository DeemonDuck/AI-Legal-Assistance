"""Offline tests for aggregation and corpus statistics.

Deliberately uses fabricated clause objects rather than real extractions, so the
maths can be verified with no API key and no cost. The statistics are the part
of this pipeline most likely to be quietly wrong -- a bad percentile produces a
plausible-looking number rather than an error -- so it is the part most worth
testing directly.

Run:  py tests/test_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_ai.corpus import build_stats  # noqa: E402
from legal_ai.profile import PERPETUAL, build_profile  # noqa: E402
from legal_ai.schemas import (  # noqa: E402
    ClauseAttributes,
    ClauseType,
    ExtractedClause,
    PartyFavored,
)

failures: list[str] = []


def check(label: str, actual, expected) -> None:
    if actual == expected:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}: expected {expected!r}, got {actual!r}")
        failures.append(label)


def clause(**attribute_overrides) -> ExtractedClause:
    """An otherwise-null clause with only the named attributes set."""
    blank = {name: None for name in ClauseAttributes.model_fields}
    blank["carve_outs_present"] = []
    blank.update(attribute_overrides)
    return ExtractedClause(
        clause_type=ClauseType.OTHER,
        heading=None,
        plain_summary="test clause",
        party_favored=PartyFavored.NEUTRAL,
        attributes=ClauseAttributes(**blank),
        aggressiveness_signals=[],
    )


print("\n--- Aggregation rules (clause -> document) ---")

# Longest duration binds, so max wins over an earlier smaller value.
p = build_profile("max", [clause(term_months=24), clause(term_months=60)])
check("term_months takes the max", p.get("term_months"), 60)

# Perpetual must beat any finite value despite being encoded as -1.
p = build_profile("perp", [clause(term_months=120), clause(term_months=PERPETUAL)])
check("perpetual beats a large finite term", p.get("term_months"), PERPETUAL)

# Shortest imposed deadline is the binding constraint.
p = build_profile("min", [clause(notice_period_days=30), clause(notice_period_days=3)])
check("notice_period_days takes the min", p.get("notice_period_days"), 3)

# One-way anywhere makes the agreement one-way in effect.
p = build_profile("mut", [clause(is_mutual=True), clause(is_mutual=False)])
check("is_mutual: false wins", p.get("is_mutual"), False)

# An obligation exists if any clause creates it.
p = build_profile("inj", [clause(injunctive_relief=False), clause(injunctive_relief=True)])
check("injunctive_relief: any true", p.get("injunctive_relief"), True)

# Carve-outs split across definition and exclusions clauses must combine.
p = build_profile("cv", [
    clause(carve_outs_present=["publicly_available"]),
    clause(carve_outs_present=["independently_developed", "publicly_available"]),
])
check("carve_outs union without duplicates",
      sorted(p.get("carve_outs_present")),
      ["independently_developed", "publicly_available"])

# "No carve-outs anywhere" is a real finding and must survive as [].
p = build_profile("none", [clause(term_months=24)])
check("empty carve_outs recorded as []", p.get("carve_outs_present"), [])

# The flagged clause must be locatable in the document.
p = build_profile("src", [clause(term_months=12), clause(term_months=48)])
check("source clause index recorded", p.source_clause.get("term_months"), 1)


print("\n--- Corpus statistics ---")

# Eight documents; terms 18/24/24/24/36/36/36/60 -> median 30.
terms = [18, 24, 24, 24, 36, 36, 36, 60]
profiles = [build_profile(f"doc{i}", [clause(term_months=t)]) for i, t in enumerate(terms)]
stats = build_stats(profiles, provenance="test")

check("n_documents", stats.n_documents, 8)
check("median term", stats.numeric["term_months"].median, 30.0)
check("p25 term", stats.numeric["term_months"].p25, 24.0)
check("min term", stats.numeric["term_months"].minimum, 18.0)
check("max term", stats.numeric["term_months"].maximum, 60.0)

# Perpetual documents must be counted but kept out of the percentile maths.
mixed = profiles + [build_profile("perp", [clause(term_months=PERPETUAL)])]
stats_mixed = build_stats(mixed, provenance="test")
check("perpetual counted", stats_mixed.numeric["term_months"].n_perpetual, 1)
check("perpetual excluded from median",
      stats_mixed.numeric["term_months"].median, 30.0)
check("n_finite excludes perpetual", stats_mixed.numeric["term_months"].n_finite, 8)

# Boolean fractions: 5 of 8 mutual.
mutual_flags = [True, True, True, True, True, False, False, False]
bool_profiles = [
    build_profile(f"d{i}", [clause(is_mutual=m)]) for i, m in enumerate(mutual_flags)
]
bool_stats = build_stats(bool_profiles, provenance="test")
check("mutual count", bool_stats.boolean["is_mutual"].n_true, 5)
check("mutual fraction", round(bool_stats.boolean["is_mutual"].true_fraction, 3), 0.625)

# Carve-out frequency across documents.
carve_profiles = [
    build_profile("a", [clause(carve_outs_present=["publicly_available"])]),
    build_profile("b", [clause(carve_outs_present=["publicly_available"])]),
    build_profile("c", [clause(carve_outs_present=[])]),
]
carve_stats = build_stats(carve_profiles, provenance="test")
check("carve-out present in 2 of 3",
      carve_stats.carve_outs.counts["publicly_available"], 2)
check("carve-out fraction",
      round(carve_stats.carve_outs.fraction("publicly_available"), 3), 0.667)
check("absent carve-out counts zero",
      carve_stats.carve_outs.counts["independently_developed"], 0)

# Small corpora must announce themselves as not authoritative.
check("small corpus flagged as not credible", stats.is_credible, False)
check("credibility note warns",
      "indicative" in stats.credibility_note(), True)

# Schema version stamping. Statistics built under one set of field descriptions
# and scored against documents read under another produce a plausible number
# with no error, which is why this is a hard gate rather than a warning.
from dataclasses import replace  # noqa: E402

from legal_ai.extract import SCHEMA_VERSION  # noqa: E402

check("stats record the current schema version", stats.schema_version, SCHEMA_VERSION)
check("matching version reports no mismatch",
      stats.schema_mismatch(SCHEMA_VERSION), None)
check("differing version reports a mismatch",
      stats.schema_mismatch("999") is not None, True)
check("mismatch message names the rebuild command",
      "build_corpus.py" in stats.schema_mismatch("999"), True)

# A statistics file written before versions were stamped loads with None, and
# must be treated as a mismatch rather than assumed current.
legacy = replace(stats, schema_version=None)
check("unstamped legacy stats are a mismatch",
      legacy.schema_mismatch(SCHEMA_VERSION) is not None, True)
check("legacy mismatch says so in words",
      "unrecorded" in legacy.schema_mismatch(SCHEMA_VERSION), True)


print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("All statistics tests passed.")
