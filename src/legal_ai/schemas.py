"""Clause taxonomy and extraction schemas -- the contract the whole pipeline reads from.

Design note (this is the load-bearing decision of the project):

  Embedding similarity answers "is this about the same topic?", NOT "is this an
  unusual deal term?". A 2-year confidentiality term and a 10-year one are ~0.95
  cosine-similar -- identical wording, one number different. So similarity cannot
  be the deviation signal.

  Instead we extract TYPED ATTRIBUTES per clause (term_months, is_mutual,
  carve_outs, ...) and score deviation by comparing those values against the
  distribution of the same attribute across the market corpus. That gives a
  defensible, explainable number: "your term is 60 months; 85% of market NDAs
  are 24-36 months."

Schema shape note: every field is REQUIRED but NULLABLE (`X | None` with no
default). This is deliberate -- the Anthropic API's strict structured-output mode
requires all fields listed in `required` with additionalProperties=false.
Nullable-required lets the model say "this clause has no term" without breaking
the schema.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ClauseType(str, Enum):
    """Clause types found in NDAs.

    OTHER is the escape hatch -- always keep it. An unmatched clause must never
    be silently dropped; it should land in OTHER and still be shown to the user.
    """

    DEFINITION_OF_CONFIDENTIAL_INFO = "definition_of_confidential_info"
    EXCLUSIONS_CARVE_OUTS = "exclusions_carve_outs"
    OBLIGATIONS_OF_RECEIVING_PARTY = "obligations_of_receiving_party"
    PERMITTED_DISCLOSURES = "permitted_disclosures"
    TERM_AND_DURATION = "term_and_duration"
    RETURN_OR_DESTRUCTION = "return_or_destruction"
    GOVERNING_LAW = "governing_law"
    JURISDICTION_AND_VENUE = "jurisdiction_and_venue"
    DISPUTE_RESOLUTION = "dispute_resolution"
    REMEDIES_AND_INJUNCTIVE_RELIEF = "remedies_and_injunctive_relief"
    INDEMNIFICATION = "indemnification"
    LIMITATION_OF_LIABILITY = "limitation_of_liability"
    NON_SOLICITATION = "non_solicitation"
    NON_COMPETE = "non_compete"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    NO_LICENSE_NO_WARRANTY = "no_license_no_warranty"
    ASSIGNMENT = "assignment"
    NOTICES = "notices"
    ENTIRE_AGREEMENT_SEVERABILITY = "entire_agreement_severability"
    OTHER = "other"


class PartyFavored(str, Enum):
    """Which side a clause tilts toward.

    In an NDA the RECEIVING party bears the obligations, so a
    receiving-party-favorable clause is friendlier to the person signing.
    """

    DISCLOSING_PARTY = "disclosing_party"
    RECEIVING_PARTY = "receiving_party"
    NEUTRAL = "neutral"


# Controlled vocabulary. Free text would make corpus comparison impossible --
# "publicly known" and "in the public domain" must land in the same bucket.
STANDARD_CARVE_OUTS = [
    "already_known_to_recipient",
    "publicly_available",
    "independently_developed",
    "rightfully_received_from_third_party",
    "required_by_law_or_court_order",
]


class ClauseAttributes(BaseModel):
    """Typed, comparable facts extracted from a single clause.

    Most fields are null for most clauses -- a governing-law clause has no
    term_months. That is expected. The scorer only compares a field across the
    clauses where it is populated.
    """

    # --- Duration -----------------------------------------------------------
    term_months: int | None = Field(
        description=(
            "Duration of the confidentiality obligation in MONTHS. Convert years "
            "to months (3 years -> 36). Use -1 for a perpetual or indefinite "
            "obligation. Null if this clause does not state a duration."
        )
    )
    survival_months: int | None = Field(
        description=(
            "How long obligations survive after termination, in months. "
            "-1 for perpetual. Null if not stated."
        )
    )
    notice_period_days: int | None = Field(
        description="Any required notice period, in days. Null if not stated."
    )

    # --- Structure ----------------------------------------------------------
    is_mutual: bool | None = Field(
        description=(
            "True if obligations are reciprocal (both parties bound), False if "
            "one-way (only one party bound). Null if not determinable."
        )
    )

    # --- Carve-outs ---------------------------------------------------------
    carve_outs_present: list[str] = Field(
        description=(
            "Which standard exclusions appear. Use ONLY these exact values: "
            + ", ".join(STANDARD_CARVE_OUTS)
            + ". Empty list if none appear. Missing carve-outs are a major red "
            "flag: an NDA without them can cover information the recipient "
            "already lawfully had."
        )
    )

    # --- Jurisdiction -------------------------------------------------------
    governing_law_jurisdiction: str | None = Field(
        description=(
            "Governing law as a short place name, e.g. 'Delaware', 'California', "
            "'England and Wales'. Null if not stated."
        )
    )
    dispute_forum: str | None = Field(
        description=(
            "One of: 'courts', 'binding_arbitration', 'mediation_then_arbitration'. "
            "Null if not stated."
        )
    )

    # --- Liability and remedies ---------------------------------------------
    injunctive_relief: bool | None = Field(
        description="True if the clause grants injunctive relief or specific performance."
    )
    liability_capped: bool | None = Field(
        description=(
            "True if liability is capped at a stated amount, False if explicitly "
            "uncapped or unlimited. Null if the clause is silent on liability."
        )
    )
    liability_cap_description: str | None = Field(
        description=(
            "The cap in the document's own words, e.g. 'fees paid in the prior 12 "
            "months'. Null if there is no cap."
        )
    )
    indemnification_present: bool | None = Field(
        description="True if the clause imposes an indemnity obligation."
    )
    attorney_fees_shifting: bool | None = Field(
        description="True if the losing party must pay the winning party's legal fees."
    )

    # --- Restrictive covenants (often smuggled into NDAs -- always surface) --
    non_solicit_months: int | None = Field(
        description="Duration of any non-solicitation restriction, in months. Null if absent."
    )
    non_compete_months: int | None = Field(
        description=(
            "Duration of any non-compete restriction, in months. Null if absent. "
            "A non-compete inside an NDA is unusual and should always be surfaced."
        )
    )

    # --- Misc obligations ---------------------------------------------------
    assignment_allowed: bool | None = Field(
        description="True if the agreement may be assigned or transferred, False if prohibited."
    )
    return_or_destroy_required: bool | None = Field(
        description="True if the recipient must return or destroy materials on request."
    )


class ExtractedClause(BaseModel):
    """One clause after LLM extraction.

    The same shape is written to disk for the uploaded document and for every
    corpus document, so the scorer can compare them directly without translation.
    """

    clause_type: ClauseType = Field(
        description="The single best-fitting clause type. Use 'other' if nothing fits."
    )
    heading: str | None = Field(
        description=(
            "The clause's heading or number as it appears in the document, e.g. "
            "'4. Term'. Null if the clause is unnumbered."
        )
    )
    plain_summary: str = Field(
        description=(
            "What this clause does, in one sentence, for a non-lawyer. No legal "
            "jargon, no hedging."
        )
    )
    party_favored: PartyFavored = Field(
        description="Which party this clause's terms favor."
    )
    attributes: ClauseAttributes = Field(
        description="Typed facts extracted from the clause text."
    )
    aggressiveness_signals: list[str] = Field(
        description=(
            "Short phrases naming anything unusually one-sided, e.g. 'perpetual "
            "obligation', 'no standard carve-outs', 'uncapped indemnity', "
            "'non-compete bundled into NDA'. Empty list if the clause reads as standard."
        )
    )


class ClauseExtraction(BaseModel):
    """Top-level response shape for one extraction call over a whole document."""

    clauses: list[ExtractedClause]


# --- Which attributes the deviation scorer actually compares -----------------
# Numeric fields -> percentile against the corpus distribution.
# Boolean fields -> frequency ("92% of market NDAs are mutual").
# These live here rather than in scoring.py so the schema and the scorer can
# never drift apart.

NUMERIC_SCORED_ATTRIBUTES = [
    "term_months",
    "survival_months",
    "non_solicit_months",
    "non_compete_months",
    # Scored with the direction INVERTED -- a shorter compliance deadline is
    # worse for the signer, not better. scoring.NUMERIC_RULES encodes that.
    "notice_period_days",
]

BOOLEAN_SCORED_ATTRIBUTES = [
    "is_mutual",
    "injunctive_relief",
    "liability_capped",
    "indemnification_present",
    "attorney_fees_shifting",
    "assignment_allowed",
    "return_or_destroy_required",
]
