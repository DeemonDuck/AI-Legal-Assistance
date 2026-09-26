"""Compare a document profile against corpus statistics and flag what deviates.

THIS STAGE MAKES NO LLM CALLS, DELIBERATELY.

Every finding here is produced by arithmetic and a sentence template. That is a
design choice, not a shortcut, and it buys three things:

  1. Reproducibility. The same document produces byte-identical findings every
     run. An LLM asked "is this clause unfair?" answers differently each time,
     which is indefensible when a user asks why the report changed.
  2. Falsifiability. Every claim is a statement about the corpus that the user
     can check -- "longer than 7 of the 8 reference NDAs" points at data, not at
     a model's opinion.
  3. Cost and latency. Flagging is free and instant; the paid LLM work is spent
     only on the clauses that survive this filter (the negotiation stage).

DIRECTION MATTERS MORE THAN DISTANCE.

An unusual value is only a problem if it is unusual in the direction that hurts
the person signing. A 6-month confidentiality term is just as far from the
median as a 60-month one, but only one of them is a risk. Every attribute
therefore declares which direction is adverse, and favourable outliers are
reported separately as leverage rather than mixed in with the warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from legal_ai.corpus import CorpusStats
from legal_ai.profile import PERPETUAL, DocumentProfile
from legal_ai.schemas import STANDARD_CARVE_OUTS

# A carve-out this common in the corpus is effectively expected; its absence is
# a finding rather than a stylistic difference.
CARVE_OUT_EXPECTED_THRESHOLD = 0.70

# Boolean features rarer than this in the corpus are treated as genuinely
# unusual rather than merely uncommon.
RARE_FRACTION = 0.25
UNCOMMON_FRACTION = 0.50


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FAVOURABLE = "favourable"  # unusual, but in the signer's favour

    @property
    def rank(self) -> int:
        return {"high": 3, "medium": 2, "low": 1, "favourable": 0}[self.value]


@dataclass(frozen=True)
class NumericRule:
    label: str
    unit: str
    higher_is_worse: bool
    # True when the term appearing AT ALL is itself the finding, because no
    # reference NDA contains it. Without this the strongest possible signal --
    # "nothing in the market does this" -- is silently discarded, since an
    # attribute absent from the whole corpus has no distribution to compare
    # against. A non-compete smuggled into an NDA is the motivating case.
    absence_in_market_is_adverse: bool = False


@dataclass(frozen=True)
class BooleanRule:
    label: str
    adverse_value: bool  # the value that is worse for the person signing
    # True when holding the adverse value is itself the finding because NO
    # reference NDA does. Same reasoning as the numeric flag of the same name:
    # an attribute absent from the whole corpus has no distribution to compare
    # against, so without this the strongest available signal is discarded.
    # Found live -- an uncapped indemnity and prevailing-party fee shifting were
    # both extracted correctly and then silently dropped for want of a baseline.
    absence_in_market_is_adverse: bool = False


# Which direction hurts the signer. In an NDA the person signing is normally the
# RECEIVING party, who carries the obligations -- so longer durations and more
# obligations are adverse, and a shorter deadline imposed on them is adverse too.
NUMERIC_RULES: dict[str, NumericRule] = {
    "term_months": NumericRule("Confidentiality term", "months", higher_is_worse=True),
    "survival_months": NumericRule("Survival period", "months", higher_is_worse=True),
    "non_solicit_months": NumericRule(
        "Non-solicitation period", "months",
        higher_is_worse=True, absence_in_market_is_adverse=True,
    ),
    "non_compete_months": NumericRule(
        "Non-compete period", "months",
        higher_is_worse=True, absence_in_market_is_adverse=True,
    ),
    # A SHORTER compliance deadline is harder to meet, so low is adverse here.
    "notice_period_days": NumericRule("Notice/compliance period", "days", higher_is_worse=False),
}

BOOLEAN_RULES: dict[str, BooleanRule] = {
    "is_mutual": BooleanRule("Mutual obligations", adverse_value=False),
    "liability_capped": BooleanRule("Liability cap", adverse_value=False),
    "indemnification_present": BooleanRule(
        "Indemnity obligation", adverse_value=True, absence_in_market_is_adverse=True,
    ),
    "attorney_fees_shifting": BooleanRule(
        "Legal-fee shifting", adverse_value=True, absence_in_market_is_adverse=True,
    ),
    "injunctive_relief": BooleanRule("Injunctive relief", adverse_value=True),
    "assignment_allowed": BooleanRule("Assignment permitted", adverse_value=False),
    "return_or_destroy_required": BooleanRule("Return/destroy obligation", adverse_value=True),
}

CARVE_OUT_LABELS = {
    "already_known_to_recipient": "information you already knew",
    "publicly_available": "publicly available information",
    "independently_developed": "information you develop independently",
    "rightfully_received_from_third_party": "information received from a third party",
    "required_by_law_or_court_order": "disclosure required by law",
}


@dataclass
class Deviation:
    """One flagged finding, ready to render without further processing."""

    attribute: str
    label: str
    severity: Severity
    document_value: object
    finding: str  # what this document says
    market: str  # what the corpus says
    why_it_matters: str
    clause_index: int | None = None

    @property
    def is_risk(self) -> bool:
        return self.severity is not Severity.FAVOURABLE


@dataclass
class DeviationReport:
    document: str
    deviations: list[Deviation] = field(default_factory=list)
    corpus_note: str = ""
    n_corpus_documents: int = 0
    skipped: list[str] = field(default_factory=list)

    @property
    def risks(self) -> list[Deviation]:
        return [d for d in self.deviations if d.is_risk]

    @property
    def favourable(self) -> list[Deviation]:
        return [d for d in self.deviations if not d.is_risk]

    def count(self, severity: Severity) -> int:
        return sum(1 for d in self.deviations if d.severity is severity)

    def headline(self) -> str:
        high, medium = self.count(Severity.HIGH), self.count(Severity.MEDIUM)
        if high:
            return f"{high} high-risk and {medium} moderate deviations from the reference corpus."
        if medium:
            return f"{medium} moderate deviations from the reference corpus. Nothing high-risk."
        if self.risks:
            return "Only minor deviations from the reference corpus."
        return "No material deviations from the reference corpus."


def _score_numeric(attribute: str, value: int, stats, clause_index: int | None) -> Deviation | None:
    rule = NUMERIC_RULES[attribute]

    # Perpetual is handled before any arithmetic: it is not a large number, it
    # is a different kind of value, and it is the single most severe finding
    # this scorer can produce when the market is finite.
    if value == PERPETUAL:
        if stats.n_finite == 0:
            return None
        return Deviation(
            attribute=attribute,
            label=rule.label,
            severity=Severity.HIGH,
            document_value=PERPETUAL,
            finding=f"{rule.label.lower()} never expires",
            market=(
                f"all {stats.n_finite} reference NDAs expire, "
                f"the longest after {stats.maximum:.0f} {rule.unit}"
            ),
            why_it_matters=(
                "An obligation with no end date binds you indefinitely. You cannot "
                "know when it is safe to use what you learned, and the restriction "
                "outlives the business relationship that justified it."
            ),
            clause_index=clause_index,
        )

    if not stats.values:
        return None

    # A corpus where every document is perpetual cannot say a finite value is
    # unusual in the adverse direction.
    exceeded = stats.rank_of(float(value))
    n = len(stats.values)
    worse_than_all = value > stats.maximum if rule.higher_is_worse else value < stats.minimum

    if rule.higher_is_worse:
        adverse = value > stats.p75
        favourable = value < stats.p25
        worst_decile = value > stats.p90
        rank_phrase = f"longer than {exceeded} of the {n} reference NDAs"
    else:
        adverse = value < stats.p25
        favourable = value > stats.p75
        # p10 is the mirror of p90 on the other tail. This used to read
        # `value < stats.minimum`, which is the SAME condition as
        # worse_than_all below -- and since that is tested first, the MEDIUM
        # branch was unreachable for every lower-is-worse attribute. Severity
        # collapsed to a cliff: a deadline shorter than 7 of the 8 reference
        # NDAs scored LOW, identical to one shorter than 2 of them.
        p10 = stats.percentile(0.10)
        worst_decile = p10 is not None and value < p10
        rank_phrase = f"shorter than {n - exceeded} of the {n} reference NDAs"

    if not adverse and not favourable:
        return None

    if favourable:
        severity = Severity.FAVOURABLE
    elif worse_than_all:
        severity = Severity.HIGH
    elif worst_decile:
        severity = Severity.MEDIUM
    else:
        severity = Severity.LOW

    market = (
        f"reference median {stats.median:.0f} {rule.unit} "
        f"(range {stats.minimum:.0f}-{stats.maximum:.0f}); {rank_phrase}"
    )

    if severity is Severity.FAVOURABLE:
        why = "This is more favourable to you than most of the reference corpus."
    elif rule.higher_is_worse:
        why = (
            f"A longer {rule.label.lower()} extends how long you carry this "
            f"obligation, and with it the window in which you can be held liable."
        )
    else:
        why = (
            f"A shorter {rule.label.lower()} gives you less time to comply, which "
            "makes an accidental breach more likely."
        )

    return Deviation(
        attribute=attribute,
        label=rule.label,
        severity=severity,
        document_value=value,
        finding=f"{value} {rule.unit}",
        market=market,
        why_it_matters=why,
        clause_index=clause_index,
    )


def _score_boolean(
    attribute: str, value: bool, stats, clause_index: int | None
) -> Deviation | None:
    rule = BOOLEAN_RULES[attribute]
    if stats.n == 0:
        return None

    # Fraction of the corpus that shares this document's value.
    share = stats.true_fraction if value else (1 - stats.true_fraction)
    matching = stats.n_true if value else stats.n - stats.n_true
    is_adverse = value == rule.adverse_value

    # Common in the corpus and therefore unremarkable, whichever way it points.
    if share > UNCOMMON_FRACTION:
        return None

    if not is_adverse:
        return Deviation(
            attribute=attribute,
            label=rule.label,
            severity=Severity.FAVOURABLE,
            document_value=value,
            finding=_phrase_boolean(rule, value),
            market=_share_phrase(matching, stats.n),
            why_it_matters="Unusual, and in your favour.",
            clause_index=clause_index,
        )

    severity = Severity.HIGH if share <= RARE_FRACTION else Severity.MEDIUM

    return Deviation(
        attribute=attribute,
        label=rule.label,
        severity=severity,
        document_value=value,
        finding=_phrase_boolean(rule, value),
        market=_share_phrase(matching, stats.n),
        why_it_matters=_boolean_rationale(attribute),
        clause_index=clause_index,
    )


def _share_phrase(matching: int, total: int) -> str:
    """'none of the 8' reads better than 'only 0 of 8', and zero is the most
    common case for genuinely unusual terms -- so it is worth special-casing."""
    if matching == 0:
        return f"none of the {total} reference NDAs do this"
    if matching == 1:
        return f"only 1 of {total} reference NDAs does this"
    return f"only {matching} of {total} reference NDAs do this"


def _phrase_boolean(rule: BooleanRule, value: bool) -> str:
    return rule.label if value else f"no {rule.label.lower()}"


def _boolean_rationale(attribute: str) -> str:
    return {
        "is_mutual": (
            "Only you carry confidentiality obligations. The other side can share "
            "your information freely while you cannot share theirs."
        ),
        "liability_capped": (
            "Your financial exposure is unlimited. A cap ties the worst case to "
            "something proportionate, usually the value of the deal."
        ),
        "indemnification_present": (
            "You may have to cover the other side's losses and legal costs, which "
            "can far exceed anything you gain from the agreement."
        ),
        "attorney_fees_shifting": (
            "If a dispute goes badly you pay their legal bill as well as your own, "
            "which raises the cost of defending yourself even when you are right."
        ),
        "injunctive_relief": (
            "They can obtain a court order against you quickly, without first "
            "proving financial loss."
        ),
        "assignment_allowed": (
            "You cannot transfer this agreement, which can complicate a sale or "
            "restructuring of your business."
        ),
        "return_or_destroy_required": (
            "You must return or destroy materials on request, which needs a process "
            "for finding every copy."
        ),
    }.get(attribute, "This term is unusual compared with the reference corpus.")


def _score_carve_outs(profile: DocumentProfile, stats: CorpusStats) -> list[Deviation]:
    present = profile.get("carve_outs_present")
    if present is None or stats.carve_outs.n == 0:
        return []

    expected = [
        term for term in STANDARD_CARVE_OUTS
        if stats.carve_outs.fraction(term) >= CARVE_OUT_EXPECTED_THRESHOLD
    ]
    missing = [term for term in expected if term not in present]
    if not missing:
        return []

    clause_index = profile.source_clause.get("carve_outs_present")

    # Missing every expected exclusion is a categorically different finding from
    # missing one, so it gets its own message rather than four separate flags.
    if len(missing) == len(expected) and len(expected) >= 3:
        return [
            Deviation(
                attribute="carve_outs_present",
                label="Standard exclusions",
                severity=Severity.HIGH,
                document_value=[],
                finding="none of the standard exclusions appear",
                market=(
                    f"all {len(expected)} appear in at least "
                    f"{int(CARVE_OUT_EXPECTED_THRESHOLD * 100)}% of reference NDAs"
                ),
                why_it_matters=(
                    "Without these exclusions the agreement can cover information you "
                    "already had, information that is public, and information you work "
                    "out yourself. That makes it very hard to prove you did not misuse "
                    "anything."
                ),
                clause_index=clause_index,
            )
        ]

    return [
        Deviation(
            attribute=f"carve_out:{term}",
            label="Missing exclusion",
            severity=Severity.MEDIUM,
            document_value=False,
            finding=f"no exclusion for {CARVE_OUT_LABELS.get(term, term)}",
            market=(
                f"present in {stats.carve_outs.counts[term]} of "
                f"{stats.carve_outs.n} reference NDAs"
            ),
            why_it_matters=(
                f"Without this exclusion, {CARVE_OUT_LABELS.get(term, term)} is treated "
                "as confidential even though it should not be."
            ),
            clause_index=clause_index,
        )
        for term in missing
    ]


def score_document(profile: DocumentProfile, stats: CorpusStats) -> DeviationReport:
    """Compare one document against the corpus and return everything that deviates.

    Deterministic: same inputs always produce the same findings, in the same order.
    """
    report = DeviationReport(
        document=profile.name,
        corpus_note=stats.credibility_note(),
        n_corpus_documents=stats.n_documents,
    )

    for attribute in NUMERIC_RULES:
        value = profile.get(attribute)
        if value is None:
            continue
        attribute_stats = stats.numeric.get(attribute)
        if attribute_stats is None:
            rule = NUMERIC_RULES[attribute]
            if rule.absence_in_market_is_adverse:
                # No reference NDA contains this term at all. That is the
                # strongest signal available, not a gap in coverage.
                report.deviations.append(
                    Deviation(
                        attribute=attribute,
                        label=rule.label,
                        severity=Severity.HIGH,
                        document_value=value,
                        finding=f"{int(value)} {rule.unit}",
                        market=(
                            f"none of the {stats.n_documents} reference NDAs "
                            "contain this restriction at all"
                        ),
                        why_it_matters=(
                            f"A {rule.label.lower()} is not a confidentiality term. "
                            "Including one here restricts what work you can take on, "
                            "which goes well beyond keeping information secret."
                        ),
                        clause_index=profile.source_clause.get(attribute),
                    )
                )
                continue
            # Otherwise there is genuinely no baseline. Recorded rather than
            # dropped: "we could not check this" is information the user needs,
            # and silence would imply "this is fine".
            report.skipped.append(attribute)
            continue
        deviation = _score_numeric(
            attribute, int(value), attribute_stats, profile.source_clause.get(attribute)
        )
        if deviation:
            report.deviations.append(deviation)

    for attribute in BOOLEAN_RULES:
        value = profile.get(attribute)
        if value is None:
            continue
        attribute_stats = stats.boolean.get(attribute)
        if attribute_stats is None:
            rule = BOOLEAN_RULES[attribute]
            if rule.absence_in_market_is_adverse and bool(value) == rule.adverse_value:
                report.deviations.append(
                    Deviation(
                        attribute=attribute,
                        label=rule.label,
                        severity=Severity.HIGH,
                        document_value=value,
                        finding=_phrase_boolean(rule, bool(value)),
                        market=(
                            f"no reference NDA imposes this "
                            f"({stats.n_documents} compared)"
                        ),
                        why_it_matters=_boolean_rationale(attribute),
                        clause_index=profile.source_clause.get(attribute),
                    )
                )
                continue
            report.skipped.append(attribute)
            continue
        deviation = _score_boolean(
            attribute, bool(value), attribute_stats, profile.source_clause.get(attribute)
        )
        if deviation:
            report.deviations.append(deviation)

    report.deviations.extend(_score_carve_outs(profile, stats))

    # Most severe first; stable tie-break on attribute name so ordering is
    # reproducible across runs rather than dependent on dict iteration.
    report.deviations.sort(key=lambda d: (-d.severity.rank, d.attribute))
    return report
