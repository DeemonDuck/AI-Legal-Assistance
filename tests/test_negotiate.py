"""Offline tests for the negotiation layer.

The LLM's output quality cannot be tested without calling it. What CAN be tested
offline is everything around the call, which is where the silent failures live:
cache keys that collide, prompts missing the grounding data, and the no-risk
short circuit that should avoid spending a request at all.

Run:  py tests/test_negotiate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_ai.negotiate import (  # noqa: E402
    ClauseNegotiation,
    NegotiationResult,
    NegotiationSet,
    RiskLevel,
    _cache_path,
    _format_deviations,
    draft_email,
    negotiate,
)
from legal_ai.parsing import RawClause  # noqa: E402
from legal_ai.scoring import Deviation, DeviationReport, Severity  # noqa: E402

failures: list[str] = []


def check(label: str, actual, expected) -> None:
    if actual == expected:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}: expected {expected!r}, got {actual!r}")
        failures.append(label)


def deviation(attribute="term_months", severity=Severity.HIGH, clause_index=None):
    return Deviation(
        attribute=attribute, label="Confidentiality term", severity=severity,
        document_value=-1, finding="never expires",
        market="all 8 reference NDAs expire, the longest after 60 months",
        why_it_matters="Binds you indefinitely.", clause_index=clause_index,
    )


CLAUSES = [
    RawClause(index=0, heading="1. Definitions", text="Definitions text."),
    RawClause(index=1, heading="3. Term", text="Obligations continue in perpetuity."),
]


print("\n--- No-risk short circuit ---")

# A clean document must not trigger an API call at all. Since no key is set in
# this environment, reaching the client would raise -- so returning normally
# proves the short circuit fired.
clean = DeviationReport(document="clean", deviations=[])
result = negotiate(clean, CLAUSES)
check("clean report returns without calling the API", result.negotiations, [])
check("clean report is not marked as cached", result.from_cache, False)
check("clean report still explains itself",
      "no specific term flagged" in result.overall_assessment, True)

# Favourable-only findings are not risks and must also short circuit.
favourable = DeviationReport(
    document="fav",
    deviations=[deviation(severity=Severity.FAVOURABLE)],
)
check("favourable-only report short circuits too",
      negotiate(favourable, CLAUSES).negotiations, [])


print("\n--- Prompt grounding ---")

report = DeviationReport(
    document="d", deviations=[deviation(clause_index=1)],
    corpus_note="Compared against 8 reference NDAs.",
)
prompt = _format_deviations(report, CLAUSES)

check("prompt carries the attribute id", "term_months" in prompt, True)
check("prompt carries the corpus comparison",
      "all 8 reference NDAs expire" in prompt, True)
check("prompt carries the statistical severity", "high" in prompt, True)
# A redline written without the original wording reads like it came from a
# different agreement, so the clause text has to reach the model.
check("prompt includes the source clause text",
      "in perpetuity" in prompt, True)
check("prompt names the clause heading", "3. Term" in prompt, True)

# Favourable findings must not be argued about -- they are leverage, not asks.
mixed = DeviationReport(document="d", deviations=[
    deviation(attribute="term_months", severity=Severity.HIGH),
    deviation(attribute="notice_period_days", severity=Severity.FAVOURABLE),
])
prompt = _format_deviations(mixed, CLAUSES)
check("favourable findings excluded from the prompt",
      "notice_period_days" in prompt, False)


print("\n--- Cache keys ---")

a = _cache_path("payload one")
b = _cache_path("payload two")
check("different payloads produce different cache files", a == b, False)
check("same payload is stable across calls", _cache_path("payload one"), a)
# The email is derived from the same negotiations but is a different artefact;
# sharing a key would serve an email where positions were requested.
check("email cache key differs from negotiation key",
      _cache_path("email|payload one") == a, False)
check("cache files land in the cache directory", a.parent.name, ".cache")


print("\n--- Schemas ---")

NegotiationSet.model_json_schema()
check("NegotiationSet schema builds", True, True)
# attribute + risk_level + 6 written fields.
check("ClauseNegotiation has all eight fields",
      len(ClauseNegotiation.model_fields), 8)
for field in ("counterparty_justification", "fallback_position", "suggested_redline"):
    check(f"schema requires {field}", field in ClauseNegotiation.model_fields, True)


print("\n--- Risk vs rarity disagreement ---")

def negotiation(attribute, risk):
    return ClauseNegotiation(
        attribute=attribute, risk_level=risk, plain_explanation="x",
        counterparty_justification="x", your_position="x",
        suggested_redline="x", fallback_position="x", talking_point="x",
    )

report = DeviationReport(document="d", deviations=[
    deviation(attribute="term_months", severity=Severity.HIGH),
    deviation(attribute="is_mutual", severity=Severity.MEDIUM),
    deviation(attribute="notice_period_days", severity=Severity.FAVOURABLE),
])
result = NegotiationResult(
    negotiations=[
        negotiation("term_months", RiskLevel.HIGH),      # agrees
        negotiation("is_mutual", RiskLevel.LOW),         # disagrees
        negotiation("notice_period_days", RiskLevel.HIGH),  # favourable, ignored
    ],
    overall_assessment="", from_cache=False,
)
gaps = result.disagreements(report)
check("agreement is not reported as a gap",
      any(g[0] == "term_months" for g in gaps), False)
check("disagreement is reported",
      ("is_mutual", "medium", "low") in gaps, True)
check("favourable findings excluded from gap analysis",
      any(g[0] == "notice_period_days" for g in gaps), False)


print("\n--- Email short circuit ---")

empty = NegotiationResult(negotiations=[], overall_assessment="", from_cache=False)
check("no negotiations means no email API call",
      "nothing to ask for" in draft_email(empty), True)


print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("All negotiation tests passed.")
