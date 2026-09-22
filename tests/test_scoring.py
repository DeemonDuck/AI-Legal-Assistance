"""Offline tests for the deviation scorer.

Builds a synthetic market corpus and scores documents against it with no API
key and no cost. The scorer is pure arithmetic over profiles and stats, which is
exactly why it can be tested this thoroughly -- and why it should be, since a
mis-signed comparison produces a confident wrong answer rather than an error.

Run:  py tests/test_scoring.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from legal_ai.corpus import build_stats  # noqa: E402
from legal_ai.profile import PERPETUAL, build_profile  # noqa: E402
from legal_ai.scoring import Severity, score_document  # noqa: E402
from test_stats import clause  # noqa: E402

failures: list[str] = []


def check(label: str, actual, expected) -> None:
    if actual == expected:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}: expected {expected!r}, got {actual!r}")
        failures.append(label)


ALL_CARVE_OUTS = [
    "already_known_to_recipient",
    "publicly_available",
    "independently_developed",
    "rightfully_received_from_third_party",
    "required_by_law_or_court_order",
]

# A market where: terms run 18-60 months (median 30), 6 of 8 are mutual,
# every document has all five carve-outs, none has a non-compete,
# 1 of 8 shifts legal fees, notice periods run 14-60 days.
MARKET = [
    (24, True, 30), (36, True, 30), (36, True, 30), (60, True, 60),
    (24, True, 14), (18, True, 30), (24, False, 30), (36, False, 30),
]
market_profiles = [
    build_profile(
        f"market{i}",
        [clause(
            term_months=t, is_mutual=m, notice_period_days=n,
            carve_outs_present=ALL_CARVE_OUTS,
            attorney_fees_shifting=(i == 0),
            liability_capped=True,
        )],
    )
    for i, (t, m, n) in enumerate(MARKET)
]
STATS = build_stats(market_profiles, provenance="test market")


def score_one(**attributes):
    profile = build_profile("subject", [clause(**attributes)])
    return score_document(profile, STATS)


def find(report, attribute):
    return next((d for d in report.deviations if d.attribute == attribute), None)


print("\n--- Schema / scorer consistency ---")

# This caught a real bug: notice_period_days had a scoring rule but was absent
# from NUMERIC_SCORED_ATTRIBUTES, so the corpus never computed stats for it and
# the scorer silently skipped it. No error, no finding -- the attribute simply
# was never checked. Pinning the invariant means the next added attribute cannot
# repeat it.
from legal_ai.scoring import BOOLEAN_RULES, NUMERIC_RULES  # noqa: E402
from legal_ai.schemas import (  # noqa: E402
    BOOLEAN_SCORED_ATTRIBUTES,
    NUMERIC_SCORED_ATTRIBUTES,
)

check("every numeric rule has corpus stats computed for it",
      sorted(set(NUMERIC_RULES) - set(NUMERIC_SCORED_ATTRIBUTES)), [])
check("every boolean rule has corpus stats computed for it",
      sorted(set(BOOLEAN_RULES) - set(BOOLEAN_SCORED_ATTRIBUTES)), [])


print("\n--- Numeric: durations ---")

r = score_one(term_months=24, carve_outs_present=ALL_CARVE_OUTS)
check("median-ish term is not flagged", find(r, "term_months"), None)

r = score_one(term_months=PERPETUAL, carve_outs_present=ALL_CARVE_OUTS)
d = find(r, "term_months")
check("perpetual term flagged HIGH", d.severity, Severity.HIGH)
check("perpetual phrased as never expiring", "never expires" in d.finding, True)

r = score_one(term_months=120, carve_outs_present=ALL_CARVE_OUTS)
check("term beyond corpus max is HIGH", find(r, "term_months").severity, Severity.HIGH)

r = score_one(term_months=48, carve_outs_present=ALL_CARVE_OUTS)
check("term above p75 but within range is flagged",
      find(r, "term_months").severity in (Severity.LOW, Severity.MEDIUM), True)

r = score_one(term_months=6, carve_outs_present=ALL_CARVE_OUTS)
d = find(r, "term_months")
check("short term is FAVOURABLE, not a risk", d.severity, Severity.FAVOURABLE)
check("favourable finding excluded from risks", d in r.risks, False)

# Direction must invert for deadlines imposed on the signer.
r = score_one(notice_period_days=3, carve_outs_present=ALL_CARVE_OUTS)
d = find(r, "notice_period_days")
check("SHORT notice period is adverse (direction inverted)",
      d is not None and d.severity is not Severity.FAVOURABLE, True)

r = score_one(notice_period_days=90, carve_outs_present=ALL_CARVE_OUTS)
d = find(r, "notice_period_days")
check("LONG notice period is favourable", d.severity, Severity.FAVOURABLE)


print("\n--- Numeric: rank phrasing ---")

r = score_one(term_months=48, carve_outs_present=ALL_CARVE_OUTS)
check("rank statement cites the corpus size",
      "of the 8 reference NDAs" in find(r, "term_months").market, True)


print("\n--- Boolean features ---")

# 6 of 8 market NDAs are mutual, so one-way is the minority and adverse.
r = score_one(is_mutual=False, carve_outs_present=ALL_CARVE_OUTS)
d = find(r, "is_mutual")
check("one-way flagged when market is mostly mutual", d is not None, True)
check("one-way severity is HIGH", d.severity, Severity.HIGH)

r = score_one(is_mutual=True, carve_outs_present=ALL_CARVE_OUTS)
check("mutual not flagged when market is mutual", find(r, "is_mutual"), None)

# Only 1 of 8 market NDAs shifts fees, so doing it is rare and adverse.
r = score_one(attorney_fees_shifting=True, carve_outs_present=ALL_CARVE_OUTS)
check("rare adverse boolean flagged HIGH",
      find(r, "attorney_fees_shifting").severity, Severity.HIGH)

r = score_one(attorney_fees_shifting=False, carve_outs_present=ALL_CARVE_OUTS)
check("common favourable boolean not flagged",
      find(r, "attorney_fees_shifting"), None)

# All market NDAs cap liability; not capping is both rare and adverse.
r = score_one(liability_capped=False, carve_outs_present=ALL_CARVE_OUTS)
check("uncapped liability flagged HIGH",
      find(r, "liability_capped").severity, Severity.HIGH)


print("\n--- Carve-outs ---")

r = score_one(carve_outs_present=[])
d = find(r, "carve_outs_present")
check("missing ALL carve-outs is one HIGH finding", d.severity, Severity.HIGH)
# Note the colon: "carve_outs_present" is the aggregate finding and must not be
# counted here -- only the per-term "carve_out:<name>" flags, which should be
# absent because the aggregate replaced them.
check("not split into one flag per carve-out",
      len([x for x in r.deviations if x.attribute.startswith("carve_out:")]), 0)

r = score_one(carve_outs_present=ALL_CARVE_OUTS[:4])
missing = [x for x in r.deviations if x.attribute.startswith("carve_out:")]
check("one missing carve-out yields one finding", len(missing), 1)
check("missing carve-out is MEDIUM", missing[0].severity, Severity.MEDIUM)

r = score_one(carve_outs_present=ALL_CARVE_OUTS)
check("complete carve-outs produce no finding",
      [x for x in r.deviations if "carve_out" in x.attribute], [])


print("\n--- Report assembly ---")

r = score_one(
    term_months=PERPETUAL, is_mutual=False, carve_outs_present=[],
    attorney_fees_shifting=True, liability_capped=False,
)
check("all five aggressive terms detected", len(r.risks) >= 5, True)
check("sorted most severe first",
      [d.severity.rank for d in r.deviations] == sorted(
          (d.severity.rank for d in r.deviations), reverse=True), True)
check("headline names high-risk count", "high-risk" in r.headline(), True)
check("corpus credibility note carried into report",
      "reference NDAs" in r.corpus_note, True)

# Determinism: identical inputs must produce identical output ordering.
again = score_one(
    term_months=PERPETUAL, is_mutual=False, carve_outs_present=[],
    attorney_fees_shifting=True, liability_capped=False,
)
check("scoring is deterministic",
      [d.attribute for d in r.deviations], [d.attribute for d in again.deviations])

# A restrictive covenant absent from the ENTIRE corpus is the strongest signal
# available, not a coverage gap. This previously fell into `skipped` and the
# worst term in a document was silently dropped from the report -- the preview
# run is what exposed it.
r = score_one(non_compete_months=24, carve_outs_present=ALL_CARVE_OUTS)
d = find(r, "non_compete_months")
check("non-compete absent from market is flagged, not skipped", d is not None, True)
check("absent-from-market severity is HIGH", d.severity, Severity.HIGH)
check("absent-from-market phrasing says none contain it",
      "none of the" in d.market, True)
check("absent-from-market not also listed as skipped",
      "non_compete_months" in r.skipped, False)

# But an attribute with no baseline AND no adverse-by-absence meaning must still
# be reported as unchecked -- silence would imply "this is fine".
r = score_one(survival_months=36, carve_outs_present=ALL_CARVE_OUTS)
check("attribute with no baseline recorded as skipped",
      "survival_months" in r.skipped, True)
check("attribute with no baseline produces no invented finding",
      find(r, "survival_months"), None)

# A clean document should say so rather than manufacturing findings.
r = score_one(term_months=30, is_mutual=True, carve_outs_present=ALL_CARVE_OUTS,
              liability_capped=True, attorney_fees_shifting=False)
check("clean document yields no risks", r.risks, [])
check("clean headline says no material deviations",
      "No material deviations" in r.headline(), True)


print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("All scoring tests passed.")
