"""Document loading and clause splitting: PDF/DOCX/TXT -> list[RawClause].

Deliberate scope limit: text-layer PDFs only. No OCR, no scanned documents.
OCR would add a heavy dependency and a long tail of quality problems for a
feature no demo needs. If a scanned PDF is uploaded, `load_document` raises a
message saying so rather than silently returning garbage -- silent garbage here
would poison every downstream stage and be very hard to trace back.

Splitting strategy is regex-first with a paragraph fallback:

  1. Find heading lines (numbered sections, ARTICLE headings, ALL-CAPS lines).
  2. Start a new clause at each heading.
  3. If that yields implausibly few clauses, fall back to blank-line paragraphs.

This is intentionally not an LLM call. Clause splitting is high-volume, cheap
to get approximately right with regex, and an LLM pass here would add latency
and cost to every document for marginal gain. The extraction stage sees the
clause text anyway and can recover from a slightly wrong boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# The one list of file types this project can read. Everything that needs to
# know -- the corpus collector, the eval's document finder, the uploader widget
# -- imports this rather than restating it, because four copies meant adding a
# format required finding all four and missing one failed inconsistently.
#
# A tuple, not a set, because evaluate.py resolves a bare document name by
# trying suffixes in order and that order should be stable rather than
# whatever a set happens to iterate.
SUPPORTED_SUFFIXES: tuple[str, ...] = (".txt", ".md", ".docx", ".pdf")

# A PDF page that yields almost no text is the classic scanned-document signal.
_MIN_CHARS_PER_PAGE_FOR_TEXT_LAYER = 50

# Below this, a "clause" is almost certainly a stray heading, page number, or
# splitting artifact rather than real content.
_MIN_CLAUSE_CHARS = 40

# If heading-based splitting finds fewer than this, the document probably isn't
# conventionally numbered and we should fall back to paragraphs.
_MIN_PLAUSIBLE_CLAUSE_COUNT = 3


@dataclass
class RawClause:
    """One clause before any LLM involvement.

    `index` is positional order in the document and is what the UI uses to show
    clauses in reading order, so it must stay stable through the pipeline.
    """

    index: int
    heading: str | None
    text: str

    @property
    def full_text(self) -> str:
        """Heading plus body, which is what gets sent for extraction -- the
        heading is often the strongest signal of clause type."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


# --- Loading ----------------------------------------------------------------


def load_document(path: Path) -> str:
    """Read a document to plain text. Raises ValueError with an actionable
    message on unsupported or unreadable input."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}. "
            "Scanned PDFs are not supported (no OCR)."
        )

    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n".join(pages)

    # Detect a scanned PDF and fail loudly rather than returning near-empty text.
    if pages and len(text.strip()) < _MIN_CHARS_PER_PAGE_FOR_TEXT_LAYER * len(pages):
        raise ValueError(
            f"'{path.name}' appears to be a scanned PDF with no text layer "
            f"({len(text.strip())} characters across {len(pages)} pages). "
            "OCR is out of scope. Convert it to DOCX or text and retry."
        )
    return text


def _load_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs]

    # Tables often carry real terms (notice addresses, fee schedules). Dropping
    # them silently loses content, so flatten them into the text stream.
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts)


# --- Cleaning ---------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Strip PDF extraction artifacts that would otherwise become fake clauses."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        # Bare page numbers, and "Page 3 of 11" style footers.
        if re.fullmatch(r"\d{1,3}", stripped):
            continue
        if re.fullmatch(r"(?i)page\s+\d+\s*(of\s*\d+)?", stripped):
            continue
        lines.append(stripped)

    cleaned = "\n".join(lines)
    # Collapse runs of blank lines to exactly one, so paragraph splitting is stable.
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


# --- Heading detection ------------------------------------------------------

# Ordered by specificity. Each must match at the START of a line.
_HEADING_PATTERNS = [
    # "ARTICLE IV", "Article 4 - Confidentiality"
    re.compile(r"^(ARTICLE|Article)\s+([IVXLC]+|\d+)\b.*$"),
    # "Section 3.", "SECTION 3.2 Confidentiality"
    re.compile(r"^(SECTION|Section)\s+\d+(\.\d+)*\.?\b.*$"),
    # "1.", "1.1", "4.2.1 Confidential Information" -- the common NDA form.
    re.compile(r"^\d+(\.\d+)*\.?\s+\S.*$"),
    # "(a) The Receiving Party shall..." at line start.
    re.compile(r"^\([a-z0-9]{1,3}\)\s+\S.*$"),
]

# A short, fully-capitalised line is nearly always a heading ("CONFIDENTIALITY").
_ALLCAPS_HEADING = re.compile(r"^[A-Z][A-Z\s,&'\-\.]{2,60}$")


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(p.match(stripped) for p in _HEADING_PATTERNS):
        return True
    # Require at least two letters so "IV." or "A." alone doesn't qualify.
    if _ALLCAPS_HEADING.match(stripped) and len(re.findall(r"[A-Z]", stripped)) >= 2:
        return True
    return False


# Words a heading title may leave lowercase. Everything else has to be
# capitalised, and that is what separates a title from a short sentence.
_TITLE_STOPWORDS = frozenset("a an and as at by for in of on or the to with".split())

_TITLE_MAX_CHARS = 60


def _looks_like_title(remainder: str) -> bool:
    """Whether a heading's remainder is a section title rather than prose.

    Real filings write `1. Definitions.` and `8. No Warranty; No Licence` --
    a title that ends the line, sometimes closed with a full stop, sometimes
    carrying an internal semicolon. Both fell through every rule above and
    left the heading as a bare `1.`, stranding the title in the body. Nothing
    was lost for extraction, which reads the body anyway, but every citation
    pointing at that clause lost the one word telling a reader where to look
    -- and showing someone where a finding came from is most of what makes it
    checkable. Six of the fourteen clauses in a real SEC-filed NDA hit this.

    The discriminator is CAPITALISATION, not length. A title capitalises every
    word except short connectives; a sentence does not. That is what keeps
    "Return or Destruction of Materials" apart from "This Agreement is
    governed by Delaware law", which is just as short. Accepting the latter
    would hand back a clause with an empty body, and _merge_stubs would then
    fold it into its neighbour -- silently gluing two real clauses together,
    which is a worse failure than the one being fixed.
    """
    text = remainder.strip()
    stripped = text.rstrip(".").strip()

    if not stripped or len(stripped) > _TITLE_MAX_CHARS:
        return False

    # A trailing colon introduces a list. The words before it are a lead-in
    # ("the Parties agree as follows:"), not the name of a section.
    if text.endswith(":"):
        return False

    # More than one sentence is prose, however it is capitalised.
    if re.search(r"\.\s+\S", text):
        return False

    words = [w for w in re.split(r"[\s;,/&-]+", stripped) if w]
    return bool(words) and all(
        word.lower() in _TITLE_STOPWORDS or word[0].isupper() or word[0].isdigit()
        for word in words
    )


def _split_heading_from_body(heading_line: str) -> tuple[str, str]:
    """A heading line often carries the clause's first sentence, e.g.
    '5. Term. This Agreement shall remain in effect for two years.'
    Keep the label as the heading and push the rest into the body."""
    match = re.match(
        r"^((?:ARTICLE|Article|SECTION|Section)?\s*[\dIVXLC\.\(\)a-z]+\.?)\s+(.*)$",
        heading_line.strip(),
    )
    if not match:
        return heading_line.strip(), ""

    label, remainder = match.group(1).strip(), match.group(2).strip()

    if not remainder:
        return label, ""

    # "5. Term. This Agreement shall..." -> heading "5. Term", body the rest.
    #
    # The character class is deliberately broad -- anything up to the first
    # full stop or colon. Real filings title sections "Termination; Duration
    # of Obligations." and "Waivers; Amendments; Assignment; Counterparts.",
    # and close some with a colon rather than a stop ("Non-Publicity: All
    # media releases..."). The previous pattern allowed only letters, spaces,
    # hyphens and apostrophes across at most 41 characters, so every one of
    # those fell through to a bare label.
    #
    # Breadth is safe because _looks_like_title then has to agree it is a
    # title and not the opening sentence of the clause. Matching widely and
    # validating is more honest than encoding the judgement into the regex,
    # where it cannot be read.
    title_match = re.match(r"^([A-Z][^.:]{0,70}?)[.:]\s+(\S.*)$", remainder)
    if title_match and _looks_like_title(title_match.group(1)):
        return f"{label} {title_match.group(1).strip()}", title_match.group(2).strip()

    # "5. Term" alone on its line -> the whole remainder is the title.
    # Detected by being short and carrying no sentence-ending punctuation.
    # Without this, headings degrade to bare numbers ("5.") and the clause title
    # -- the single strongest signal of clause type -- gets buried in the body.
    if len(remainder) <= 60 and not re.search(r"[.;:]", remainder):
        return f"{label} {remainder}", ""

    # "1. Definitions." / "8. No Warranty; No Licence" -- a title that closes
    # the line. Checked last so it only sees lines every rule above rejected,
    # which until now returned a bare label. The trailing full stop is dropped
    # because it belongs to the line, not to the title.
    if _looks_like_title(remainder):
        return f"{label} {remainder.rstrip('.').strip()}", ""

    return label, remainder


# --- Splitting --------------------------------------------------------------


def split_into_clauses(text: str) -> list[RawClause]:
    """Split document text into clauses. Never returns an empty list for
    non-empty input -- worst case the whole document comes back as one clause,
    which the extractor can still work with."""
    cleaned = _clean_text(text)
    if not cleaned:
        return []

    clauses = _split_by_headings(cleaned)

    if len(clauses) < _MIN_PLAUSIBLE_CLAUSE_COUNT:
        paragraph_clauses = _split_by_paragraphs(cleaned)
        if len(paragraph_clauses) > len(clauses):
            clauses = paragraph_clauses

    if not clauses:
        clauses = [RawClause(index=0, heading=None, text=cleaned)]

    return _renumber(clauses)


def _split_by_headings(text: str) -> list[RawClause]:
    clauses: list[RawClause] = []
    current_heading: str | None = None
    current_body: list[str] = []

    def flush() -> None:
        body = "\n".join(current_body).strip()
        if current_heading is None and not body:
            return
        clauses.append(RawClause(index=len(clauses), heading=current_heading, text=body))

    for line in text.splitlines():
        if _is_heading(line):
            if current_heading is not None or current_body:
                flush()
            current_heading, remainder = _split_heading_from_body(line)
            current_body = [remainder] if remainder else []
        else:
            current_body.append(line)

    if current_heading is not None or current_body:
        flush()

    return _merge_stubs(clauses)


def _split_by_paragraphs(text: str) -> list[RawClause]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    clauses = [RawClause(index=i, heading=None, text=b) for i, b in enumerate(blocks)]
    return _merge_stubs(clauses)


def _merge_stubs(clauses: list[RawClause]) -> list[RawClause]:
    """Fold too-short clauses into the following one.

    A standalone heading with no body ("CONFIDENTIALITY" on its own line above
    a numbered list) would otherwise become a contentless clause and waste an
    extraction slot -- and, worse, show up as an empty row in the final report.
    """
    merged: list[RawClause] = []
    pending: RawClause | None = None

    for clause in clauses:
        if pending is not None:
            heading = pending.heading or clause.heading
            # When the stub supplied the heading, the follower's own heading is
            # still content and must come through in its body.
            tail = clause.full_text if pending.heading else clause.text
            body = "\n".join(p for p in (pending.text, tail) if p).strip()
            clause = RawClause(index=pending.index, heading=heading, text=body)
            pending = None

        if len(clause.text.strip()) < _MIN_CLAUSE_CHARS:
            pending = clause
            continue

        merged.append(clause)

    # A trailing stub has nothing to merge into; keep it rather than drop content.
    if pending is not None:
        if merged:
            last = merged[-1]
            merged[-1] = RawClause(
                index=last.index,
                heading=last.heading,
                text=f"{last.text}\n{pending.full_text}".strip(),
            )
        else:
            merged.append(pending)

    return merged


def _renumber(clauses: list[RawClause]) -> list[RawClause]:
    return [RawClause(index=i, heading=c.heading, text=c.text) for i, c in enumerate(clauses)]


def parse_document(path: Path) -> list[RawClause]:
    """Load and split in one step. This is the entry point the rest of the
    pipeline should call."""
    return split_into_clauses(load_document(Path(path)))
