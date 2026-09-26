"""Clause text -> typed attributes, via one structured-output call per document.

This is the stage that replaces the embedding step from the original plan. The
model classifies clause type AND extracts typed attributes in a single pass,
because it is reading the clause either way -- asking for `clause_type` in the
same response costs nothing extra and removes the need for an embedding model,
a vector store, and ~2GB of dependencies.

Two implementation decisions that matter:

ONE CALL PER DOCUMENT, not per clause. Clauses are interdependent -- whether an
NDA is mutual, whether carve-outs appear anywhere, and whether a term in section
3 is modified by section 9 are all cross-clause facts. Per-clause calls would
each be blind to that context and would multiply latency by ~12x.

RESULTS ARE CACHED TO DISK by content hash. Extraction is the slow, paid step,
and during development the same documents get re-run constantly. The cache makes
iteration free and makes a live demo instant rather than a 40-second wait.
Delete .cache/ to force re-extraction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from legal_ai import config
from legal_ai.llm import LLMError, structured
from legal_ai.parsing import RawClause, parse_document
from legal_ai.schemas import ClauseExtraction, ExtractedClause

# Bump when schemas.py changes shape, so cached results from an older schema are
# not silently reused and mistaken for current output.
SCHEMA_VERSION = "3"

# Strict structured output requires every attribute on every clause, so a clause
# emits all 16 fields even when 14 are null.
#
# MEASURED on gpt-oss-120b at reasoning_effort=low: 1278 completion tokens for
# 4 clauses. But reasoning spend varies widely between documents -- 283 to 1433
# tokens on same-sized inputs -- so a value fitted to one measurement truncates
# on denser documents, which is exactly what happened on the first paced run.
#
# 480 is deliberately above the measured average, and _extract_batch() halves
# and retries anything that still overflows. Sizing for the common case with a
# fallback beats sizing for the worst case and wasting the budget every call.
#
# Documents are therefore split into batches that fit the output ceiling rather
# than being sent whole and truncated. Truncation is the dangerous failure here:
# it looks like a successful analysis of a shorter document. Batching costs some
# cross-clause context, which is a real loss (see the module docstring) but a
# smaller one than silently losing half a document.
TOKENS_PER_CLAUSE = 480
BATCH_SAFETY_FACTOR = 0.8


def _batch_size() -> int:
    """How many clauses fit in one response, given the output ceiling."""
    usable = int(config.MAX_TOKENS * BATCH_SAFETY_FACTOR)
    return max(1, usable // TOKENS_PER_CLAUSE)

SYSTEM_PROMPT = """You are a contracts analyst extracting structured data from NDA clauses.

You are building a dataset for comparing this document against market-standard \
NDAs. Your job is extraction and classification, not advice.

Rules:

1. Extract only what the text actually says. If a clause does not state a value, \
the corresponding attribute is null. Never infer a "typical" value to fill a gap \
-- a null is data, a guess is contamination.

2. Convert all durations to the requested unit. Years become months (3 years -> \
36). Spelled-out numbers become integers ("thirty-six (36) months" -> 36). Use \
-1 for perpetual or indefinite obligations; do not use a large number.

3. carve_outs_present uses the controlled vocabulary exactly as given. If a \
clause describes an exclusion in different words, map it to the matching term. \
If it describes no exclusions, return an empty list.

4. Judge party_favored from the substance, not the title. An agreement labelled \
"Mutual" whose obligations run one way favors the disclosing party.

5. aggressiveness_signals should name specific, checkable features -- "perpetual \
obligation", "no standard carve-outs", "non-compete bundled into NDA". Do not \
include generic commentary, and return an empty list for clauses that read as \
standard.

6. plain_summary is one sentence a non-lawyer can act on. No jargon, no hedging, \
no restating the clause in legal language.

Return one entry per clause given, in the same order.

Return a JSON OBJECT with a single key "clauses" whose value is the array. Do \
not return a bare array as the top-level value."""


@dataclass
class ExtractionResult:
    """Extraction output paired with the raw clauses it came from.

    Kept together because the report needs original clause text alongside the
    extracted attributes, and re-deriving the pairing later invites off-by-one
    bugs when the model returns a different number of clauses than was sent.
    """

    raw_clauses: list[RawClause]
    extracted: list[ExtractedClause]
    from_cache: bool

    def pairs(self) -> list[tuple[RawClause, ExtractedClause]]:
        # strict=True on purpose. extract_clauses() truncates both sides to the
        # same length before constructing this, so a mismatch here means that
        # guarantee broke -- and a silent zip would then pair clause text with
        # a different clause's attributes and report it as fact.
        return list(zip(self.raw_clauses, self.extracted, strict=True))


def _cache_path(clauses: list[RawClause]) -> Path:
    """Cache key covers clause text, model, and schema version -- changing any
    of them must produce a different key, or stale output gets served."""
    payload = json.dumps(
        {
            "clauses": [c.full_text for c in clauses],
            "model": config.EXTRACTION_MODEL,
            "schema_version": SCHEMA_VERSION,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return config.CACHE_DIR / f"extract_{digest}.json"


def _format_clauses(clauses: list[RawClause]) -> str:
    blocks = []
    for clause in clauses:
        label = clause.heading or f"(clause {clause.index + 1}, no heading)"
        blocks.append(f"--- CLAUSE {clause.index + 1} | {label} ---\n{clause.text}")
    return "\n\n".join(blocks)


def _extract_batch(batch: list[RawClause]) -> list[ExtractedClause]:
    """Extract one batch, halving and retrying if the response is truncated.

    A fixed batch size cannot be right for every document. Output length depends
    on how dense the clauses are and on how many reasoning tokens the model
    decides to spend -- measured between 283 and 1433 on identical-sized inputs.
    Tuning the batch to the worst case would waste most of the budget on the
    common case, and tuning it to the average truncates the dense documents.

    So the batch size is set for the common case and shrinks on demand. A
    truncated response is detected, the batch is split in half, and each half is
    retried. Cost is one wasted call on the documents that need it, instead of a
    failed run or a permanently oversized ceiling.
    """
    user_content = (
        f"Extract structured data from the following {len(batch)} NDA clauses."
        f"\n\n{_format_clauses(batch)}"
    )
    try:
        return list(
            structured(
                SYSTEM_PROMPT, user_content, ClauseExtraction,
                model=config.EXTRACTION_MODEL,
            ).clauses
        )
    except LLMError as exc:
        truncated = "truncated" in str(exc)
        if truncated and len(batch) > 1:
            middle = len(batch) // 2
            print(
                f"  ! {len(batch)} clauses exceeded the output cap; "
                f"splitting into {middle} + {len(batch) - middle}"
            )
            return _extract_batch(batch[:middle]) + _extract_batch(batch[middle:])

        if truncated:
            # A single clause that will not fit is a real limit, not variance.
            raise RuntimeError(
                "A single clause exceeded the output cap. Raise MAX_TOKENS in "
                "config.py -- and check RATE_LIMIT, since a larger ceiling also "
                "reserves more of the per-minute budget."
            ) from exc
        raise RuntimeError(f"Extraction failed: {exc}") from exc


def extract_clauses(clauses: list[RawClause], *, use_cache: bool = True) -> ExtractionResult:
    """Extract typed attributes for every clause in one API call.

    Raises RuntimeError with an actionable message on API failure; callers are
    expected to surface that to the user rather than continue with partial data.
    """
    if not clauses:
        return ExtractionResult(raw_clauses=[], extracted=[], from_cache=False)

    cache_file = _cache_path(clauses)
    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return ExtractionResult(
            raw_clauses=clauses,
            extracted=[ExtractedClause.model_validate(c) for c in cached["clauses"]],
            from_cache=True,
        )

    size = _batch_size()
    batches = [clauses[i:i + size] for i in range(0, len(clauses), size)]

    extracted: list[ExtractedClause] = []
    for batch in batches:
        extracted.extend(_extract_batch(batch))

    # The model can return a different count than was sent. Align rather than
    # crash -- a truncated report is recoverable, a hard failure mid-demo is not
    # -- but never silently mispair clause text with someone else's attributes.
    if len(extracted) != len(clauses):
        n = min(len(extracted), len(clauses))
        print(
            f"  ! extraction returned {len(extracted)} clauses for {len(clauses)} sent; "
            f"using first {n}"
        )
        clauses, extracted = clauses[:n], extracted[:n]

    cache_file.write_text(
        json.dumps({"clauses": [c.model_dump(mode="json") for c in extracted]}, indent=2),
        encoding="utf-8",
    )

    return ExtractionResult(raw_clauses=clauses, extracted=extracted, from_cache=False)


def extract_document(path: Path, *, use_cache: bool = True) -> ExtractionResult:
    """Parse and extract in one step -- the entry point for both the uploaded
    document and every corpus document, so both sides of the comparison are
    produced by identical code."""
    return extract_clauses(parse_document(Path(path)), use_cache=use_cache)
