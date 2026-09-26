"""Offline tests for document loading and clause splitting.

Parsing was the last stage in the pipeline with no tests, and it is the one
where a fault is hardest to see. Every later stage is handed whatever this
module produces and has no way to tell a badly-cut clause from a real one: a
term buried in the tail of the previous clause still extracts, still profiles,
still scores, and still prints a confident report -- against a document that was
never read properly. The parser is also the only stage that touches untrusted
input directly.

Everything here runs on strings and temp files. No API key, no cost.

Run:  py tests/test_parsing.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_ai.parsing import (  # noqa: E402
    RawClause,
    _clean_text,
    _is_heading,
    _split_heading_from_body,
    load_document,
    parse_document,
    split_into_clauses,
)

failures: list[str] = []


def check(label: str, actual, expected) -> None:
    if actual == expected:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}: expected {expected!r}, got {actual!r}")
        failures.append(label)


def body(n: int = 60) -> str:
    """Filler long enough to clear the _MIN_CLAUSE_CHARS stub threshold."""
    return "The Receiving Party shall keep the information confidential. " * (n // 60 + 1)


print("\n--- Heading detection ---")

check("numbered heading", _is_heading("3. Confidentiality"), True)
check("multi-level numbered heading", _is_heading("4.2.1 Confidential Information"), True)
check("ARTICLE heading", _is_heading("ARTICLE IV"), True)
check("Section heading", _is_heading("Section 3.2 Term"), True)
check("lettered sub-clause", _is_heading("(a) The Receiving Party shall comply."), True)
check("ALL CAPS heading", _is_heading("CONFIDENTIALITY"), True)

check("ordinary prose is not a heading",
      _is_heading("The Receiving Party shall keep it secret."), False)
check("blank line is not a heading", _is_heading("   "), False)
# Guards the >=2-letter rule: a bare roman numeral or initial is an enumerator,
# not a section title, and treating it as one shatters the document.
check("single-letter capital is not a heading", _is_heading("A."), False)


print("\n--- Heading / body separation ---")

# The clause title is the strongest single signal of clause type, so it has to
# stay on the heading rather than being swallowed into the body.
check("title kept with label, sentence pushed to body",
      _split_heading_from_body("5. Term. This Agreement shall remain in effect."),
      ("5. Term", "This Agreement shall remain in effect."))
check("bare title stays whole",
      _split_heading_from_body("5. Term"), ("5. Term", ""))
check("label alone survives", _split_heading_from_body("5."), ("5.", ""))


print("\n--- Heading styles real filings actually use ---")

# Everything below was taken from a Confidentiality and Non-Disclosure
# Agreement filed with the SEC. Six of its fourteen clauses used to come back
# with a bare label like "6.", stranding the title in the body -- extraction
# still read it, but every citation lost the word telling a reader where to
# look, and pointing at the source is most of what makes a finding checkable.

# A title that closes the line with a full stop.
check("title ending in a full stop",
      _split_heading_from_body("1. Definitions."), ("1. Definitions", ""))

# Semicolons are ordinary inside contract section titles. The old pattern
# allowed only letters, spaces, hyphens and apostrophes.
check("semicolons inside a title",
      _split_heading_from_body(
          "6. Termination; Duration of Obligations. Unless sooner terminated, "
          "this Agreement continues."),
      ("6. Termination; Duration of Obligations",
       "Unless sooner terminated, this Agreement continues."))
check("several semicolons",
      _split_heading_from_body(
          "8. Waivers; Amendments; Assignment; Counterparts. This Agreement "
          "may not be modified except in writing.")[0],
      "8. Waivers; Amendments; Assignment; Counterparts")

# Some sections close the title with a colon instead of a stop.
check("title closed by a colon",
      _split_heading_from_body("11. Non-Publicity: All media releases shall be "
                               "approved in advance."),
      ("11. Non-Publicity", "All media releases shall be approved in advance."))

# The old cap was 41 characters, which real titles exceed routinely.
check("title longer than the old 41-character cap",
      _split_heading_from_body(
          "3. Exceptions to the Confidentiality and Non-Use Obligations. The "
          "obligations imposed by Section 2 shall not apply.")[0],
      "3. Exceptions to the Confidentiality and Non-Use Obligations")

# --- and the things that must NOT be mistaken for titles -------------------

# Prose that happens to be short. Accepting this would return an empty body,
# and _merge_stubs would then fold the clause into its neighbour -- gluing two
# real clauses together, which is worse than the bug being fixed.
check("a short sentence is not a title",
      _split_heading_from_body("12. This Agreement is governed by Delaware law."),
      ("12.", "This Agreement is governed by Delaware law."))
check("the sentence stays in the body where the scorer can read it",
      "Delaware" in _split_heading_from_body(
          "12. This Agreement is governed by Delaware law.")[1], True)

# A trailing colon introduces a list; the words before it are a lead-in.
check("a lead-in ending in a colon is not a title",
      _split_heading_from_body(
          "2. In consideration of the foregoing, the Parties agree as follows:"),
      ("2.", "In consideration of the foregoing, the Parties agree as follows:"))

# The behaviour that already worked must be unchanged.
check("title plus first sentence still splits",
      _split_heading_from_body("5. Term. This Agreement shall remain in effect."),
      ("5. Term", "This Agreement shall remain in effect."))


print("\n--- Cleaning ---")

check("bare page numbers dropped", "7" in _clean_text("Clause text\n7\nMore text"), False)
check("'Page 3 of 11' footers dropped",
      "Page 3 of 11" in _clean_text("Clause text\nPage 3 of 11\nMore"), False)
check("real content survives cleaning",
      "Clause text" in _clean_text("Clause text\n7\nMore text"), True)
# Three or more blank lines collapse to one, so paragraph splitting is stable.
check("blank-line runs collapsed", "\n\n\n" in _clean_text("A\n\n\n\n\nB"), False)


print("\n--- Splitting ---")

document = f"""1. Definitions
{body()}

2. Confidentiality
{body()}

3. Term
{body()}
"""
clauses = split_into_clauses(document)
check("numbered document splits into three clauses", len(clauses), 3)
check("headings are captured", [c.heading for c in clauses],
      ["1. Definitions", "2. Confidentiality", "3. Term"])
check("indices are positional and start at zero",
      [c.index for c in clauses], [0, 1, 2])

# An unnumbered document has no headings to split on, so the paragraph fallback
# has to carry it -- otherwise the whole document arrives as one clause and
# every finding cites "the document" instead of a place in it.
unnumbered = f"{body()}\n\n{body()}\n\n{body()}\n\n{body()}"
fallback = split_into_clauses(unnumbered)
check("paragraph fallback splits an unnumbered document", len(fallback), 4)
check("fallback clauses carry no heading",
      all(c.heading is None for c in fallback), True)

# A heading with no body of its own must not become a contentless clause; it
# folds into the clause beneath it, taking its title along.
stubbed = f"CONFIDENTIALITY\n\n1. Obligations\n{body()}\n\n2. Term\n{body()}"
merged = split_into_clauses(stubbed)
check("standalone heading does not become an empty clause",
      all(len(c.text.strip()) > 0 for c in merged), True)
check("stub content is not silently dropped",
      any("CONFIDENTIALITY" in (c.heading or "") + c.text for c in merged), True)

check("empty input yields no clauses", split_into_clauses(""), [])
check("whitespace-only input yields no clauses", split_into_clauses("   \n\n  \t "), [])

# The contract the docstring promises: non-empty input always produces at least
# one clause. Worst case the extractor sees the whole document at once, which
# still works -- returning nothing would strand the caller with no error.
check("short unstructured input still yields a clause",
      len(split_into_clauses("Keep it secret.")), 1)

# Indices must be contiguous after merging, because the report uses them to
# index back into raw_clauses -- a gap there cites the wrong clause.
for name, produced in (("numbered", clauses), ("fallback", fallback), ("merged", merged)):
    check(f"{name} indices are contiguous",
          [c.index for c in produced], list(range(len(produced))))


print("\n--- RawClause ---")

check("full_text joins heading and body",
      RawClause(0, "1. Term", "Two years.").full_text, "1. Term\nTwo years.")
check("full_text is the body alone when unheaded",
      RawClause(0, None, "Two years.").full_text, "Two years.")


print("\n--- Loading ---")

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)

    txt = root / "nda.txt"
    txt.write_text(f"1. Term\n{body()}", encoding="utf-8")
    check("txt loads", "Term" in load_document(txt), True)
    check("parse_document is load plus split", len(parse_document(txt)), 1)

    # Unsupported and missing files must fail with a message a user can act on,
    # not a stack trace from inside a parser library.
    odd = root / "contract.rtf"
    odd.write_text("whatever", encoding="utf-8")
    try:
        load_document(odd)
        check("unsupported suffix rejected", "no error", "ValueError")
    except ValueError as exc:
        check("unsupported suffix rejected", True, True)
        check("rejection names the supported types", ".docx" in str(exc), True)

    try:
        load_document(root / "absent.txt")
        check("missing file rejected", "no error", "ValueError")
    except ValueError as exc:
        check("missing file rejected", "File not found" in str(exc), True)

    # Undecodable bytes are replaced rather than raising: a mildly mis-encoded
    # document should still be reviewable.
    rough = root / "rough.txt"
    rough.write_bytes(b"1. Term\n\xff\xfe invalid bytes here\n" + body().encode())
    check("undecodable bytes do not crash the loader",
          "Term" in load_document(rough), True)


print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("All parsing tests passed.")
