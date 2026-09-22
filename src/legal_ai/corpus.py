"""Build and persist the market-standard distributions the scorer compares against.

Runs the same parse -> extract -> profile pipeline over every document in
data/corpus/, then summarises each attribute across those document profiles.

HANDLING OF PERPETUAL VALUES

A perpetual obligation is stored as -1, and it is deliberately excluded from
percentile maths and counted separately. Mixing "infinite" into a numeric
distribution corrupts the median in a way that is hard to see: with -1 treated
as a number, two perpetual NDAs would drag the median DOWN, making perpetual
terms look shorter than 24-month ones. Reporting "2 of 8 are perpetual" next to
"median finite term 30 months" is both correct and more useful to a reader.

CORPUS PROVENANCE IS THE WEAK POINT OF THIS PROJECT

Every claim the product makes is only as good as what sits in data/corpus/. The
seed templates shipped with the repo are synthetic reference drafts, not sampled
real-world agreements. The statistics below are computed correctly over whatever
is provided -- but "correctly computed" is not the same as "representative of
the market". CorpusStats therefore carries `provenance` and `n_documents` so
that any report built on it can state honestly what it is comparing against.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from legal_ai import config
from legal_ai.extract import extract_document
from legal_ai.profile import PERPETUAL, DocumentProfile, build_profile
from legal_ai.schemas import (
    BOOLEAN_SCORED_ATTRIBUTES,
    NUMERIC_SCORED_ATTRIBUTES,
    STANDARD_CARVE_OUTS,
)

CORPUS_STATS_FILE = config.CORPUS_INDEX_DIR / "corpus_stats.json"

# Below this, quartiles are noise rather than signal. Not enforced -- the corpus
# still builds -- but every consumer should surface the warning to the user.
MIN_DOCUMENTS_FOR_CREDIBLE_STATS = 15


@dataclass
class NumericStats:
    """Distribution of one numeric attribute across documents.

    Percentiles cover FINITE values only; `perpetual_count` is reported
    alongside. See the module docstring for why they are kept apart.
    """

    n: int
    n_perpetual: int
    minimum: float | None
    p25: float | None
    median: float | None
    p75: float | None
    p90: float | None
    maximum: float | None

    @property
    def n_finite(self) -> int:
        return self.n - self.n_perpetual


@dataclass
class BooleanStats:
    n: int
    n_true: int

    @property
    def true_fraction(self) -> float:
        return self.n_true / self.n if self.n else 0.0


@dataclass
class FrequencyStats:
    """Counts for categorical values (jurisdictions, dispute forums, carve-outs)."""

    n: int
    counts: dict[str, int] = field(default_factory=dict)

    def fraction(self, key: str) -> float:
        return self.counts.get(key, 0) / self.n if self.n else 0.0


@dataclass
class CorpusStats:
    n_documents: int
    built_on: str
    provenance: str
    documents: list[str]
    numeric: dict[str, NumericStats]
    boolean: dict[str, BooleanStats]
    carve_outs: FrequencyStats
    jurisdictions: FrequencyStats
    dispute_forums: FrequencyStats

    @property
    def is_credible(self) -> bool:
        return self.n_documents >= MIN_DOCUMENTS_FOR_CREDIBLE_STATS

    def credibility_note(self) -> str:
        """The sentence any report should carry. Phrased so it can be shown to a
        user verbatim -- overclaiming here is the main integrity risk."""
        if self.is_credible:
            return f"Compared against {self.n_documents} reference NDAs."
        return (
            f"Compared against only {self.n_documents} reference NDAs "
            f"(fewer than {MIN_DOCUMENTS_FOR_CREDIBLE_STATS}); "
            "treat percentile comparisons as indicative, not authoritative."
        )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile.

    Hand-rolled rather than numpy purely so the arithmetic is visible in review;
    at n<100 the performance difference is irrelevant.
    """
    if not sorted_values:
        raise ValueError("percentile of an empty list")
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _numeric_stats(values: list[int]) -> NumericStats:
    perpetual = [v for v in values if v == PERPETUAL]
    finite = sorted(float(v) for v in values if v != PERPETUAL)

    if not finite:
        return NumericStats(
            n=len(values), n_perpetual=len(perpetual),
            minimum=None, p25=None, median=None, p75=None, p90=None, maximum=None,
        )

    return NumericStats(
        n=len(values),
        n_perpetual=len(perpetual),
        minimum=finite[0],
        p25=_percentile(finite, 0.25),
        median=_percentile(finite, 0.50),
        p75=_percentile(finite, 0.75),
        p90=_percentile(finite, 0.90),
        maximum=finite[-1],
    )


def build_stats(profiles: list[DocumentProfile], *, provenance: str) -> CorpusStats:
    """Summarise document profiles into comparable distributions.

    Pure function over profiles -- no file or network access -- so the statistics
    can be unit-tested with fabricated profiles and no API key.
    """
    numeric: dict[str, NumericStats] = {}
    for attribute in NUMERIC_SCORED_ATTRIBUTES:
        values = [p.get(attribute) for p in profiles if p.get(attribute) is not None]
        if values:
            numeric[attribute] = _numeric_stats([int(v) for v in values])

    boolean: dict[str, BooleanStats] = {}
    for attribute in BOOLEAN_SCORED_ATTRIBUTES:
        values = [p.get(attribute) for p in profiles if p.get(attribute) is not None]
        if values:
            boolean[attribute] = BooleanStats(
                n=len(values), n_true=sum(1 for v in values if v is True)
            )

    # Carve-outs are counted per term: how many documents include each one.
    # A document missing a carve-out that 8/8 others have is a strong signal.
    carve_counts = {term: 0 for term in STANDARD_CARVE_OUTS}
    documents_with_carve_data = 0
    for profile in profiles:
        present = profile.get("carve_outs_present")
        if present is None:
            continue
        documents_with_carve_data += 1
        for term in present:
            if term in carve_counts:
                carve_counts[term] += 1

    def _frequency(attribute: str) -> FrequencyStats:
        counts: dict[str, int] = {}
        total = 0
        for profile in profiles:
            value = profile.get(attribute)
            if value is None:
                continue
            total += 1
            counts[str(value)] = counts.get(str(value), 0) + 1
        return FrequencyStats(n=total, counts=counts)

    return CorpusStats(
        n_documents=len(profiles),
        built_on=date.today().isoformat(),
        provenance=provenance,
        documents=[p.name for p in profiles],
        numeric=numeric,
        boolean=boolean,
        carve_outs=FrequencyStats(n=documents_with_carve_data, counts=carve_counts),
        jurisdictions=_frequency("governing_law_jurisdiction"),
        dispute_forums=_frequency("dispute_forum"),
    )


def build_corpus(
    corpus_dir: Path | None = None,
    *,
    provenance: str = "synthetic reference drafts shipped with the repository",
    use_cache: bool = True,
) -> CorpusStats:
    """Run the full pipeline over every document in the corpus directory.

    Makes one API call per uncached document. Raises RuntimeError if the
    directory is empty, since silently producing empty statistics would make
    every later deviation score meaningless without any visible error.
    """
    directory = Path(corpus_dir) if corpus_dir else config.CORPUS_DIR
    paths = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}
    )

    if not paths:
        raise RuntimeError(
            f"No corpus documents found in {directory}. "
            "The deviation scorer has nothing to compare against without them."
        )

    profiles: list[DocumentProfile] = []
    for path in paths:
        result = extract_document(path, use_cache=use_cache)
        source = "cache" if result.from_cache else "API"
        print(f"  {path.name:38} {len(result.extracted):3} clauses  ({source})")
        profiles.append(build_profile(path.stem, result.extracted))

    return build_stats(profiles, provenance=provenance)


# --- Persistence -------------------------------------------------------------


def save_stats(stats: CorpusStats, path: Path | None = None) -> Path:
    target = Path(path) if path else CORPUS_STATS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
    return target


def load_stats(path: Path | None = None) -> CorpusStats:
    target = Path(path) if path else CORPUS_STATS_FILE
    if not target.exists():
        raise RuntimeError(
            f"No corpus statistics at {target}. Run:  py build_corpus.py"
        )

    raw = json.loads(target.read_text(encoding="utf-8"))
    return CorpusStats(
        n_documents=raw["n_documents"],
        built_on=raw["built_on"],
        provenance=raw["provenance"],
        documents=raw["documents"],
        numeric={k: NumericStats(**v) for k, v in raw["numeric"].items()},
        boolean={k: BooleanStats(**v) for k, v in raw["boolean"].items()},
        carve_outs=FrequencyStats(**raw["carve_outs"]),
        jurisdictions=FrequencyStats(**raw["jurisdictions"]),
        dispute_forums=FrequencyStats(**raw["dispute_forums"]),
    )
