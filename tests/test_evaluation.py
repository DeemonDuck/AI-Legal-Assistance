"""Offline tests for the evaluation harness.

An eval that cannot fail is worse than no eval -- it produces a number that
looks like evidence while proving nothing. So the most important assertions here
feed the harness a deliberately BROKEN scorer output and check that it reports
failure. If those pass, the eval has teeth.

Run:  py tests/test_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_ai.evaluation import (  # noqa: E402
    DocumentExpectation,
    EvalResult,
    Expectation,
    evaluate_report,
    load_expectations,
)
from legal_ai.scoring import Deviation, DeviationReport, Severity  # noqa: E402

failures: list[str] = []


def check(label: str, actual, expected) -> None:
    if actual == expected:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}: expected {expected!r}, got {actual!r}")
        failures.append(label)


def deviation(attribute: str, severity: Severity) -> Deviation:
    return Deviation(
        attribute=attribute, label=attribute, severity=severity,
        document_value=None, finding="x", market="y", why_it_matters="z",
    )


def report(*deviations) -> DeviationReport:
    return DeviationReport(document="t", deviations=list(deviations))


EXPECT_TWO = DocumentExpectation(
    name="t", description="", max_high=None,
    must_flag=[
        Expectation("term_months", "high"),
        Expectation("is_mutual", "medium"),
    ],
)


print("\n--- Detection ---")

r = evaluate_report(
    report(deviation("term_months", Severity.HIGH),
           deviation("is_mutual", Severity.HIGH)),
    EXPECT_TWO,
)
check("both planted terms found", sorted(r.found), ["is_mutual", "term_months"])
check("recall 100%", r.recall, 1.0)
check("document passes", r.passed, True)

# The assertion that gives the eval teeth: a scorer that finds nothing must FAIL.
r = evaluate_report(report(), EXPECT_TWO)
check("empty report is a failure", r.passed, False)
check("both terms recorded as missed", len(r.missed), 2)
check("recall 0%", r.recall, 0.0)

r = evaluate_report(report(deviation("term_months", Severity.HIGH)), EXPECT_TWO)
check("partial detection recorded", len(r.missed), 1)
check("partial detection fails", r.passed, False)
check("recall 50%", r.recall, 0.5)


print("\n--- Severity floor ---")

# Detecting a perpetual term but calling it LOW is not a pass: severity is what
# decides whether a user acts on the finding.
r = evaluate_report(
    report(deviation("term_months", Severity.LOW),
           deviation("is_mutual", Severity.HIGH)),
    EXPECT_TWO,
)
check("under-severity is not counted as found", "term_months" in r.found, False)
check("under-severity recorded separately", len(r.under_severity), 1)
check("under-severity fails the document", r.passed, False)
check("under-severity is not a miss", len(r.missed), 0)

# Exceeding the required severity is fine.
r = evaluate_report(
    report(deviation("term_months", Severity.HIGH),
           deviation("is_mutual", Severity.HIGH)),
    EXPECT_TWO,
)
check("exceeding min severity passes", r.passed, True)


print("\n--- False positives on the clean control ---")

CONTROL = DocumentExpectation(
    name="clean", description="", max_high=0, must_flag=[],
)

r = evaluate_report(report(), CONTROL)
check("clean report passes the control", r.passed, True)
check("no false positives", r.n_high, 0)

r = evaluate_report(report(deviation("term_months", Severity.HIGH)), CONTROL)
check("HIGH on the control is a failure", r.passed, False)
check("over budget detected", r.over_high_budget, True)
check("unexpected HIGH named", r.unexpected_high, ["term_months"])

# A flag-everything scorer must not pass by having perfect recall.
BUDGETED = DocumentExpectation(
    name="mixed", description="", max_high=1,
    must_flag=[Expectation("term_months", "low")],
)
r = evaluate_report(
    report(deviation("term_months", Severity.HIGH),
           deviation("is_mutual", Severity.HIGH),
           deviation("liability_capped", Severity.HIGH)),
    BUDGETED,
)
check("perfect recall does not excuse over-flagging", r.passed, False)
check("recall still 100%", r.recall, 1.0)
check("budget overrun counted", r.n_high > r.max_high, True)


print("\n--- Aggregate scoring ---")

total = EvalResult(documents=[
    evaluate_report(
        report(deviation("term_months", Severity.HIGH),
               deviation("is_mutual", Severity.HIGH)),
        EXPECT_TWO,
    ),
    evaluate_report(report(deviation("x", Severity.HIGH)), CONTROL),
])
check("aggregate counts expectations", total.total_expected, 2)
check("aggregate counts detections", total.total_found, 2)
check("aggregate recall", total.recall, 1.0)
check("aggregate counts false positives", total.false_positives, 1)
check("one failing document fails the suite", total.passed, False)
check("headline reports both numbers",
      "Recall 2/2" in total.headline() and "1 false positive" in total.headline(),
      True)


print("\n--- Ground-truth file ---")

expectations = load_expectations()
check("_README key excluded from documents", "_README" in expectations, False)
check("all three golden documents present", len(expectations), 3)
check("control declares a zero HIGH budget",
      expectations["clean_nda_02"].max_high, 0)
check("control plants nothing",
      expectations["clean_nda_02"].must_flag, [])
check("aggressive document plants eight terms",
      len(expectations["aggressive_nda_01"].must_flag), 8)
check("perpetual term required at HIGH",
      next(e.min_severity for e in expectations["aggressive_nda_01"].must_flag
           if e.attribute == "term_months"), "high")
check("mixed document expects the individual carve-out, not the aggregate",
      any(e.attribute == "carve_out:independently_developed"
          for e in expectations["mixed_nda_03"].must_flag), True)


print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("All evaluation-harness tests passed.")
