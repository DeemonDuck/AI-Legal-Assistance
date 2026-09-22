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
  -> both sides of each flagged term are argued in one structured call,
     producing risk level + plain explanation + suggested redline + fallback
```

Three stages, three different tools, deliberately:

| Stage | How | Why |
|---|---|---|
| Extraction | LLM, structured output | Reading unstructured legal prose into typed facts is the part that needs a model |
| Scoring | Pure arithmetic, no LLM | Reproducible and falsifiable — same document, same findings, every run |
| Negotiation | LLM, structured output | Writing arguments and redlines is generative work |

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

## Quick start

```bash
py -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY
py build_corpus.py            # build market baselines (8 API calls, then cached)
streamlit run app.py          # or use the CLI below
```

## Usage

```bash
# Check clause boundaries without spending anything (no API call)
py analyze.py --parse-only data/golden/aggressive_nda_01.txt

# Full extraction
py analyze.py data/golden/aggressive_nda_01.txt

# Ignore the cache and re-call the API
py analyze.py --no-cache data/my_nda.pdf

# Build the market-standard distributions the scorer compares against
py build_corpus.py
py build_corpus.py --show     # print saved stats without rebuilding

# Negotiating positions for each flagged term (1 extra API call)
py analyze.py --negotiate data/golden/aggressive_nda_01.txt

# ...and a draft negotiation email
py analyze.py --email data/golden/aggressive_nda_01.txt

# Measure the scorer against hand-written ground truth
py evaluate.py

# Run the offline tests (no API key needed)
py tests/test_stats.py
py tests/test_scoring.py
py tests/test_evaluation.py
```

## How we know it works

`py evaluate.py` scores three documents whose correct answers were written by
reading them, not by recording what the scorer produced:

| Document | Purpose |
|---|---|
| `aggressive_nda_01` | Eight planted aggressive terms — the recall test |
| `clean_nda_02` | Deliberately unremarkable — any HIGH finding is a false positive |
| `mixed_nda_03` | Three planted terms among standard ones — tests discrimination |

The eval documents are deliberately **not** in the corpus. Scoring a document
against a baseline that contains it leaks the answer, and `check_leakage()`
enforces the separation on every run.

Extraction results are cached in `.cache/` by content hash, so re-running the
same document is instant and free. Delete `.cache/` to force re-extraction.

## Layout

| Path | What's in it |
|---|---|
| `src/legal_ai/config.py` | Paths and model IDs. All paths derive from `PROJECT_ROOT`. |
| `src/legal_ai/schemas.py` | Clause taxonomy + attribute schemas. The contract everything reads from. |
| `src/legal_ai/parsing.py` | PDF/DOCX/TXT -> clauses. Regex-based, no API calls. |
| `src/legal_ai/extract.py` | Clauses -> typed attributes via one structured-output call. |
| `src/legal_ai/profile.py` | Clause-level facts -> one document-level row. |
| `src/legal_ai/corpus.py` | Market distributions over document profiles. |
| `src/legal_ai/scoring.py` | Deviation scoring. Deterministic, no LLM calls. |
| `src/legal_ai/evaluation.py` | Measures the scorer against ground truth. |
| `analyze.py` | Analyse a single NDA. |
| `src/legal_ai/negotiate.py` | Flagged terms -> negotiating positions. The generative stage. |
| `evaluate.py` | Run the eval. |
| `app.py` | Streamlit UI. Presentation only; `src/` never imports Streamlit. |
| `build_corpus.py` | Build market statistics from `data/corpus/`. |
| `data/corpus/` | Market-standard NDA templates |
| `data/golden/` | Eval set: clean NDAs with known aggressive terms planted |

## Scope

NDAs only. The clause taxonomy and attribute schema are per-document-type, so
extending to other contract types means adding a taxonomy, not rewriting the
pipeline -- but that is explicitly out of scope here.

Not built: document drafting, model fine-tuning, jurisdiction-specific legal
reasoning, multi-document-type support.
