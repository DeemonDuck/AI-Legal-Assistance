"""Measure the scorer against hand-written ground truth.

WHY THIS EXISTS

Until this module ran, there was no evidence the scorer worked. Its output
looked plausible -- which is precisely the problem, because a deviation scorer
that flags the wrong things produces confident, well-formatted, wrong reports.
The only way to know is to feed it documents whose correct answer was decided in
advance, by reading them, and check.

TWO RULES THAT MAKE THE MEASUREMENT HONEST

1. Ground truth is written from the DOCUMENTS, never from the scorer's output.
   Recording what the scorer currently produces and calling it expected makes the
   eval a tautology: it passes at 100% no matter how wrong the scorer is, and it
   locks in today's bugs as tomorrow's requirements.

2. Eval documents are NOT in the corpus. Scoring a document against a baseline
   that includes it leaks the answer -- its own values drag the median toward
   itself, so it looks more normal than it is. data/golden/ and data/corpus/ are
   disjoint, and check_leakage() enforces that rather than trusting it.

WHAT IS MEASURED

  Recall    -- of the planted deviations, how many were found. Misses are the
               failure that matters most: a missed predatory term is the exact
               harm this tool exists to prevent.
  False     -- HIGH findings on the clean control. A tool that flags everything
  positives    has perfect recall and is useless, so this is the counterweight.
  Severity  -- whether findings met the minimum severity expected. Detecting a
               perpetual term but calling it LOW is not a pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from legal_ai import config
from legal_ai.scoring import DeviationReport, Severity

EXPECTED_FILE = config.GOLDEN_DIR / "expected.json"

_SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}


@dataclass
class Expectation:
    attribute: str
    min_severity: str
    notes: str = ""


@dataclass
class DocumentExpectation:
    name: str
    description: str
    max_high: int | None
    must_flag: list[Expectation] = field(default_factory=list)


@dataclass
class DocumentResult:
    name: str
    description: str
    found: list[str] = field(default_factory=list)
    missed: list[Expectation] = field(default_factory=list)
    under_severity: list[tuple[Expectation, Severity]] = field(default_factory=list)
    unexpected_high: list[str] = field(default_factory=list)
    n_high: int = 0
    max_high: int | None = None

    @property
    def n_expected(self) -> int:
        return len(self.found) + len(self.missed)

    @property
    def recall(self) -> float:
        return len(self.found) / self.n_expected if self.n_expected else 1.0

    @property
    def over_high_budget(self) -> bool:
        return self.max_high is not None and self.n_high > self.max_high

    @property
    def passed(self) -> bool:
        return (
            not self.missed
            and not self.under_severity
            and not self.over_high_budget
        )


@dataclass
class EvalResult:
    documents: list[DocumentResult] = field(default_factory=list)

    @property
    def total_expected(self) -> int:
        return sum(d.n_expected for d in self.documents)

    @property
    def total_found(self) -> int:
        return sum(len(d.found) for d in self.documents)

    @property
    def recall(self) -> float:
        return self.total_found / self.total_expected if self.total_expected else 1.0

    @property
    def false_positives(self) -> int:
        """HIGH findings beyond each document's budget. Only counted where a
        budget was declared -- a document with max_high null is not a control."""
        return sum(
            max(0, d.n_high - d.max_high)
            for d in self.documents
            if d.max_high is not None
        )

    @property
    def passed(self) -> bool:
        return all(d.passed for d in self.documents)

    def headline(self) -> str:
        return (
            f"Recall {self.total_found}/{self.total_expected} "
            f"({self.recall * 100:.0f}%), "
            f"{self.false_positives} false positive(s) over budget"
        )


def load_expectations(path: Path | None = None) -> dict[str, DocumentExpectation]:
    target = Path(path) if path else EXPECTED_FILE
    if not target.exists():
        raise RuntimeError(f"No ground-truth file at {target}.")

    raw = json.loads(target.read_text(encoding="utf-8"))
    expectations: dict[str, DocumentExpectation] = {}

    for name, spec in raw.items():
        if name.startswith("_"):  # _README and any other commentary keys
            continue
        expectations[name] = DocumentExpectation(
            name=name,
            description=spec.get("description", ""),
            max_high=spec.get("max_high"),
            must_flag=[
                Expectation(
                    attribute=attribute,
                    min_severity=detail.get("min_severity", "low"),
                    notes=detail.get("notes", ""),
                )
                for attribute, detail in spec.get("must_flag", {}).items()
            ],
        )
    return expectations


def evaluate_report(
    report: DeviationReport, expectation: DocumentExpectation
) -> DocumentResult:
    """Compare one scored document against its ground truth.

    Pure function -- no file or network access -- so the comparison logic can be
    tested with fabricated reports and no API key.
    """
    result = DocumentResult(
        name=expectation.name,
        description=expectation.description,
        max_high=expectation.max_high,
    )

    by_attribute = {d.attribute: d for d in report.deviations}
    result.n_high = report.count(Severity.HIGH)

    for expected in expectation.must_flag:
        found = by_attribute.get(expected.attribute)
        if found is None:
            result.missed.append(expected)
            continue

        # Detecting a perpetual term but calling it LOW is not a pass -- the
        # severity is what drives whether a user acts on it.
        actual_rank = _SEVERITY_ORDER.get(found.severity.value, 0)
        required_rank = _SEVERITY_ORDER.get(expected.min_severity, 1)
        if actual_rank < required_rank:
            result.under_severity.append((expected, found.severity))
        else:
            result.found.append(expected.attribute)

    if expectation.max_high is not None:
        planted = {e.attribute for e in expectation.must_flag}
        result.unexpected_high = sorted(
            d.attribute
            for d in report.deviations
            if d.severity is Severity.HIGH and d.attribute not in planted
        )

    return result


def check_leakage(
    corpus_dir: Path | None = None, golden_dir: Path | None = None
) -> list[str]:
    """Return the names of documents present in BOTH corpus and golden sets.

    Enforced rather than assumed: an overlap silently inflates every score, and
    it is an easy mistake to make when adding a document to round out the corpus.
    """
    corpus = Path(corpus_dir) if corpus_dir else config.CORPUS_DIR
    golden = Path(golden_dir) if golden_dir else config.GOLDEN_DIR

    suffixes = {".txt", ".md", ".pdf", ".docx"}
    corpus_names = {p.stem for p in corpus.iterdir() if p.suffix.lower() in suffixes}
    golden_names = {p.stem for p in golden.iterdir() if p.suffix.lower() in suffixes}
    return sorted(corpus_names & golden_names)
