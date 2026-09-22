"""Collapse many clause-level extractions into one document-level profile.

WHY THIS STAGE EXISTS

The unit of comparison is the DOCUMENT, not the clause. "85% of market NDAs are
mutual" is a statement about documents. If you build distributions from
clause-level values instead, a document that mentions "24 months" across three
clauses contributes three data points and skews the distribution toward itself.
So every document collapses to exactly one row first, and statistics are built
over those rows.

THE AGGREGATION RULES ARE A JUDGEMENT CALL, NOT PLUMBING

A document can state a term in more than one clause, and the values can differ.
Picking max vs min vs first is a substantive decision about what the document
"really" says, so each rule is named explicitly in ATTRIBUTE_RULES below rather
than being buried in an if-chain. The guiding principle is to take the reading
that is WORST for the person signing -- if any clause makes the agreement
one-way, it is one-way in effect; if any clause leaves liability uncapped, the
uncapped reading governs. A tool that helps someone spot risk should not round
in the optimistic direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from legal_ai.schemas import (
    BOOLEAN_SCORED_ATTRIBUTES,
    NUMERIC_SCORED_ATTRIBUTES,
    ExtractedClause,
)

# schemas.py encodes a perpetual/indefinite obligation as -1 rather than a large
# number, so "forever" can never be mistaken for a big finite value in maths.
PERPETUAL = -1


class Rule(str, Enum):
    """How to collapse several clause-level values into one document value."""

    MAX_OR_PERPETUAL = "max_or_perpetual"  # longest obligation binds; -1 dominates
    MIN = "min"  # shortest deadline binds
    ANY_TRUE = "any_true"  # obligation exists if stated anywhere
    FALSE_WINS = "false_wins"  # conservative reading governs
    UNION = "union"  # combine lists
    FIRST = "first"  # documents state this once


ATTRIBUTE_RULES: dict[str, Rule] = {
    # Durations: the longest stated obligation is the one that binds you.
    "term_months": Rule.MAX_OR_PERPETUAL,
    "survival_months": Rule.MAX_OR_PERPETUAL,
    "non_solicit_months": Rule.MAX_OR_PERPETUAL,
    "non_compete_months": Rule.MAX_OR_PERPETUAL,
    # Deadlines imposed ON you: the shortest is the binding constraint.
    "notice_period_days": Rule.MIN,
    # One-way anywhere makes the agreement one-way in effect.
    "is_mutual": Rule.FALSE_WINS,
    # An explicit "uncapped" or "may not assign" anywhere governs.
    "liability_capped": Rule.FALSE_WINS,
    "assignment_allowed": Rule.FALSE_WINS,
    # These obligations exist if any clause creates them.
    "injunctive_relief": Rule.ANY_TRUE,
    "indemnification_present": Rule.ANY_TRUE,
    "attorney_fees_shifting": Rule.ANY_TRUE,
    "return_or_destroy_required": Rule.ANY_TRUE,
    # Carve-outs are often split between the definition and exclusions clauses.
    "carve_outs_present": Rule.UNION,
    # Stated once per document.
    "governing_law_jurisdiction": Rule.FIRST,
    "dispute_forum": Rule.FIRST,
    "liability_cap_description": Rule.FIRST,
}


@dataclass
class DocumentProfile:
    """One document reduced to a single comparable row.

    `values` holds the aggregated attributes. `source_clause` records which
    clause each value came from, so the report can cite the clause that produced
    a flag -- without it, a user is told "your term is perpetual" with no way to
    find where the document says that.
    """

    name: str
    values: dict[str, object] = field(default_factory=dict)
    source_clause: dict[str, int] = field(default_factory=dict)
    n_clauses: int = 0

    def get(self, attribute: str):
        return self.values.get(attribute)

    def is_perpetual(self, attribute: str) -> bool:
        return self.values.get(attribute) == PERPETUAL


def _aggregate(attribute: str, rule: Rule, candidates: list[tuple[int, object]]):
    """Apply one rule to (clause_index, value) pairs. Returns (value, clause_index)
    or (None, None) when no clause populated the attribute."""
    if not candidates:
        return None, None

    if rule is Rule.FIRST:
        return candidates[0][1], candidates[0][0]

    if rule is Rule.UNION:
        merged: list[str] = []
        for _, value in candidates:
            for item in value or []:
                if item not in merged:
                    merged.append(item)
        return merged, candidates[0][0]

    if rule is Rule.ANY_TRUE:
        for index, value in candidates:
            if value is True:
                return True, index
        return False, candidates[0][0]

    if rule is Rule.FALSE_WINS:
        for index, value in candidates:
            if value is False:
                return False, index
        return True, candidates[0][0]

    if rule is Rule.MIN:
        index, value = min(candidates, key=lambda pair: pair[1])
        return value, index

    if rule is Rule.MAX_OR_PERPETUAL:
        # Perpetual beats any finite duration. Checked before max() because -1
        # would otherwise lose to every positive number -- the exact inversion
        # this encoding exists to avoid.
        for index, value in candidates:
            if value == PERPETUAL:
                return PERPETUAL, index
        index, value = max(candidates, key=lambda pair: pair[1])
        return value, index

    raise ValueError(f"No aggregation implemented for rule {rule} ({attribute})")


def build_profile(name: str, clauses: list[ExtractedClause]) -> DocumentProfile:
    """Collapse a document's extracted clauses into one comparable row."""
    profile = DocumentProfile(name=name, n_clauses=len(clauses))

    for attribute, rule in ATTRIBUTE_RULES.items():
        candidates: list[tuple[int, object]] = []
        for index, clause in enumerate(clauses):
            value = getattr(clause.attributes, attribute, None)
            # An empty carve-out list is a real finding ("this NDA has no
            # exclusions"), but only when no clause listed any. Skip empties
            # here and let the UNION rule produce [] if every clause was empty.
            if value is None or value == []:
                continue
            candidates.append((index, value))

        value, clause_index = _aggregate(attribute, rule, candidates)
        if value is not None:
            profile.values[attribute] = value
            if clause_index is not None:
                profile.source_clause[attribute] = clause_index
        elif rule is Rule.UNION:
            # No clause listed carve-outs at all -- record the empty list, since
            # "none present" is exactly the signal the scorer needs.
            profile.values[attribute] = []

    return profile


def profile_from_path(path: Path, clauses: list[ExtractedClause]) -> DocumentProfile:
    return build_profile(Path(path).stem, clauses)


def scored_attributes() -> list[str]:
    """Attributes the deviation scorer will compare. Sourced from schemas.py so
    the two cannot drift apart."""
    return NUMERIC_SCORED_ATTRIBUTES + BOOLEAN_SCORED_ATTRIBUTES
