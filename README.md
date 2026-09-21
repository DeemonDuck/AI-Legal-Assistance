# Legal GenAI Assistant

Helps non-lawyers understand NDAs. Upload an NDA, get a clause-by-clause report
showing which terms deviate from market-standard NDAs, why that matters in plain
language, and what to ask for instead.

**This tool provides information, not legal advice, and does not replace a lawyer.**

## How it works

```
NDA (PDF/DOCX)
  -> parse into clauses
  -> LLM extracts typed attributes per clause (term_months, is_mutual, carve_outs, ...)
  -> compare each attribute against the same attribute across a market corpus
  -> flag the outliers
  -> two personas debate each flagged clause, a synthesizer produces
     risk level + plain explanation + suggested redline
```

### Why attributes and not embedding similarity

Embedding similarity answers *"is this about the same topic?"*, not *"is this an
unusual deal term?"* A 2-year confidentiality term and a 10-year one are ~0.95
cosine-similar -- same words, one number different. Scoring on similarity would
flag oddly-worded standard clauses while missing genuinely predatory ones.

So the pipeline extracts typed attributes and compares them against the corpus
distribution instead. That produces a claim a user can check: *"your term is 60
months; 85% of market NDAs are 24-36 months."*

## Setup

```bash
py -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
cp .env.example .env    # then paste your ANTHROPIC_API_KEY into it
```

Moving between machines: see [DEVICE_SETUP.md](DEVICE_SETUP.md).

## Usage

```bash
# Check clause boundaries without spending anything (no API call)
py analyze.py --parse-only data/golden/aggressive_nda_01.txt

# Full extraction
py analyze.py data/golden/aggressive_nda_01.txt

# Ignore the cache and re-call the API
py analyze.py --no-cache data/my_nda.pdf
```

Extraction results are cached in `.cache/` by content hash, so re-running the
same document is instant and free. Delete `.cache/` to force re-extraction.

## Layout

| Path | What's in it |
|---|---|
| `src/legal_ai/config.py` | Paths and model IDs. All paths derive from `PROJECT_ROOT`. |
| `src/legal_ai/schemas.py` | Clause taxonomy + attribute schemas. The contract everything reads from. |
| `src/legal_ai/parsing.py` | PDF/DOCX/TXT -> clauses. Regex-based, no API calls. |
| `src/legal_ai/extract.py` | Clauses -> typed attributes via one structured-output call. |
| `analyze.py` | CLI entry point. |
| `data/corpus/` | Market-standard NDA templates |
| `data/golden/` | Eval set: clean NDAs with known aggressive terms planted |

## Scope

NDAs only. The clause taxonomy and attribute schema are per-document-type, so
extending to other contract types means adding a taxonomy, not rewriting the
pipeline -- but that is explicitly out of scope here.

Not built: document drafting, model fine-tuning, jurisdiction-specific legal
reasoning, multi-document-type support.
