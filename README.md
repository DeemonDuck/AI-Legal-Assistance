# NDA Reviewer

Upload an NDA. Find out which terms are unusual, why that matters, and what to
ask for instead.

> **This tool provides information, not legal advice, and it does not replace a
> lawyer.** It tells you how a document compares to others. It cannot tell you
> whether to sign.

---

## The problem

Most people asked to sign an NDA have no way to tell an ordinary agreement from
an aggressive one. Both look like the same wall of legal prose, and the terms
that matter — how long you are bound, whether the obligations run both ways,
whether standard protections were quietly removed — are single phrases buried in
paragraphs that all read alike.

A lawyer spots those in minutes because they have read hundreds of NDAs and know
what normal looks like. This tool tries to give a non-lawyer the same
comparison.

## What it produces

```
[HIGH] Confidentiality term  (see: 3. Term)
        this document : confidentiality term never expires
        reference NDAs: all 8 reference NDAs expire, the longest after 60 months
        why it matters: An obligation with no end date binds you indefinitely.
                        You cannot know when it is safe to use what you learned.

[HIGH] Non-compete period  (see: 7. Non-Competition)
        this document : 24 months
        reference NDAs: none of the 8 reference NDAs contain this restriction at all
        why it matters: A non-compete is not a confidentiality term. Including
                        one here restricts what work you can take on.
```

Then, for each flagged term, an argued negotiating position: the other side's
strongest justification, the case against it, replacement wording to ask for,
and a fallback if they refuse.

The design goal for every line of output is that **a user can check it**. Not
"this clause seems aggressive" — an opinion with nothing behind it — but "this
runs 36 months where every reference NDA uses 12", which is a claim that points
at data and can be taken into a conversation.

---

## Quick start

```bash
py -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

cp .env.example .env          # add your ANTHROPIC_API_KEY
py build_corpus.py            # build market baselines: 8 API calls, then cached
streamlit run app.py
```

Moving the project between machines: see [DEVICE_SETUP.md](DEVICE_SETUP.md).
Short version — use git, not a zip; `.venv` breaks when moved.

### Command line

```bash
# Check clause boundaries. No API call, costs nothing.
py analyze.py --parse-only path/to/nda.pdf

# Deviation report
py analyze.py path/to/nda.pdf

# ...plus negotiating positions (1 extra API call)
py analyze.py --negotiate path/to/nda.pdf

# ...plus a draft negotiation email
py analyze.py --email path/to/nda.pdf

# Corpus and evaluation
py build_corpus.py --show     # print saved market statistics
py evaluate.py                # measure the scorer against ground truth

# Offline tests — no API key needed
py tests/test_stats.py
py tests/test_scoring.py
py tests/test_evaluation.py
py tests/test_negotiate.py
```

Results are cached by document content, so re-running a document is instant and
free. Delete `.cache/` to force re-extraction.

---

## How it works

```
NDA (PDF / DOCX / TXT)
    │
    ├─ 1. Parse into clauses                     regex, no API
    │
    ├─ 2. Extract typed attributes per clause    LLM, structured output
    │     term_months, is_mutual, carve_outs, liability_capped, ...
    │
    ├─ 3. Collapse to one document-level row     pure logic
    │
    ├─ 4. Compare against the corpus             pure arithmetic
    │     flag terms that are unusual in the direction that hurts you
    │
    └─ 5. Argue both sides of each flag          LLM, structured output
          risk level, explanation, redline, fallback
```

Three stages use three different tools, and the split is deliberate:

| Stage | Tool | Why this tool |
|---|---|---|
| Extraction | LLM | Reading unstructured legal prose into typed facts is genuinely hard and genuinely what models are good at |
| Scoring | Arithmetic | Same document must produce the same findings every run, and every claim must point at checkable data |
| Negotiation | LLM | Writing arguments and contract language is generative work |

---

## Design decisions

The interesting parts of this project are the choices, so they are written down
rather than left to be inferred from the code.

### Attributes, not embedding similarity

**The original plan** was to embed each clause, retrieve the nearest
same-type clauses from a corpus, and flag low cosine similarity as "deviating
from market standard". That was abandoned, and it is the decision that shapes
everything else.

**Why it does not work.** Embedding similarity measures *topical sameness*, not
*whether a deal term is unusual*:

| Comparison | Similarity | Actually different? |
|---|---|---|
| "term of two (2) years" vs "term of ten (10) years" | ~0.95 | Enormously — five times the obligation |
| A mutual clause vs its one-way version | ~0.93 | Yes — completely different risk |
| A standard clause in unusual house style | ~0.70 | No — same terms, different words |

The failure runs both ways. A predatory 10-year term embeds almost identically
to a standard 2-year one and is **missed**; a perfectly ordinary clause in odd
phrasing scores low and is **falsely flagged**. The result is a scorer that
flags unusual *writing* and ignores unusual *terms* — and it would look
plausible in a demo, because the numbers still come out looking like numbers.

**What replaced it.** The two conflated jobs were split. The LLM classifies the
clause type *and* extracts typed attributes in the same pass — it is reading the
clause either way, so classification costs nothing extra. Deviation then becomes
ordinary statistics: where does this document's `term_months` sit in the
distribution of `term_months` across the corpus?

**What that bought:**

- A claim a user can check, instead of an uninterpretable distance score.
- A concrete handoff to the negotiation stage — it argues about a number.
- Four heavy dependencies removed: no `torch` (~2GB), no `sentence-transformers`,
  no FAISS, no vector store. A cold clone installs in about a minute.
- Explainability for free. The reason a clause was flagged *is* the comparison.

**The honest cost:** semantic retrieval is lost. A clause worded in a way the
taxonomy does not anticipate lands in `OTHER` and gets weaker treatment. NDA
clauses are highly conventional, so this is a small loss in practice — but it is
a real one.

### Scoring makes no LLM calls

Every finding is arithmetic plus a sentence template. This is a choice, not a
shortcut:

- **Reproducibility.** The same document produces identical findings every run.
  An LLM asked "is this clause unfair?" answers differently each time, which is
  indefensible when a user asks why the report changed.
- **Falsifiability.** Every claim points at corpus data rather than a model's
  opinion.
- **Cost.** Flagging is free and instant. Paid work happens only on the terms
  that survive the filter.

### Direction matters more than distance

A 6-month confidentiality term is exactly as far from the median as a 60-month
one, but only one is a risk. Every attribute declares which direction is adverse
for the person signing, and favourable outliers are reported separately as
leverage rather than mixed into the warnings.

Some attributes invert: a **shorter** compliance deadline is worse for you, not
better.

### The negotiation is one call, not a multi-turn debate

The plan called for two persona agents debating with a third synthesising —
three sequential calls per flagged clause. On a document with eight flags that
is 24 calls and 30–60 seconds of latency, and multi-turn debate tends toward
mush as each round regresses to the mean of the two positions.

The schema instead forces both stances and the synthesis into a single
structured response, requiring the model to state the counterparty's
justification **before** arguing against it. That ordering is what makes the
output usable: a position that cannot articulate the other side's reasoning is a
complaint, not a negotiating position.

### Two risk ratings that are allowed to disagree

The scorer measures how **unusual** a term is. The negotiation stage judges how
**dangerous** it is. These are different questions — uncommon terms can be
harmless, and common ones can be serious. Where the two disagree, the UI shows
the gap rather than hiding it.

### The document is the unit of comparison, not the clause

"Most NDAs are mutual" is a claim about documents. If distributions were built
from clause-level values, a document stating "24 months" across three clauses
would contribute three data points and skew the baseline toward itself. Each
document collapses to one row first.

Where clauses conflict, aggregation takes the reading **worst for the person
signing** — one-way anywhere means one-way in effect; an explicit "uncapped"
anywhere governs. A tool for spotting risk should not round optimistically.

### No OCR

Text-layer PDFs and DOCX only. A scanned PDF raises an explanatory error rather
than returning near-empty text, because silent garbage at the parsing stage
poisons everything downstream and is very hard to trace back.

---

## How we know it works

`py evaluate.py` scores three documents whose correct answers were written in
advance:

| Document | Purpose |
|---|---|
| `aggressive_nda_01` | Eight planted aggressive terms — measures recall |
| `clean_nda_02` | Deliberately unremarkable — any HIGH finding is a false positive |
| `mixed_nda_03` | Three planted terms among standard ones — measures discrimination |

The third document matters more than it looks. **A scorer that flags everything
has perfect recall and is useless**, so the clean control and a cap on HIGH
findings are the counterweight.

Two rules keep the measurement honest:

1. **Ground truth was written by reading the documents**, never by recording
   what the scorer produced. Recording current output as "expected" makes the
   eval a tautology that passes however wrong the scorer is, and locks today's
   bugs in as tomorrow's requirements.
2. **Eval documents are not in the corpus.** Scoring a document against a
   baseline containing it leaks the answer — its own values pull the median
   toward itself. `check_leakage()` enforces this on every run.

The eval also tests itself: `tests/test_evaluation.py` feeds the harness
deliberately broken scorer output and requires it to report failure. An eval
that cannot fail is worse than no eval.

---

## Known limitations

Stated plainly, because a tool that makes claims about legal documents should be
clear about what it does not know.

**The reference corpus is the weakest link.** The eight seed documents are
*synthetic reference drafts*, not sampled real-world agreements. Every
comparison is only as good as what sits in `data/corpus/`. The code phrases its
own claims accordingly — `credibility_note()` downgrades the language below 15
documents — but until real templates replace them, the honest reading is
"compared against our reference corpus", not "compared against the market".
See [data/corpus/README.md](data/corpus/README.md) for how to swap them in.

**Eight documents is too few for real percentiles.** Quartiles over eight points
are noisy. Fifteen to twenty is the minimum for the numbers to carry weight.

**No jurisdiction-specific reasoning.** A 24-month non-compete is unenforceable
in California and routine in Texas. The tool compares governing law by frequency
but knows nothing about what any jurisdiction actually enforces.

**Extraction quality is unverified at scale.** It has been checked against a
handful of documents, not measured across many. A wrong attribute value produces
a confident, well-formatted, wrong finding.

**NDAs only.** Other contract types will parse, but the taxonomy and attributes
will not fit and most clauses will land in `OTHER`.

**No OCR**, so scanned documents are rejected.

---

## What would make this better

Roughly in order of how much each would improve the result per unit of work:

| Improvement | Why it matters |
|---|---|
| **Replace the corpus with real NDAs** | Single highest-value change. SEC EDGAR exhibits are real, current, public record. Converts every claim from "our reference set" to a defensible statement about the market. |
| **Grow the corpus past 15–20 documents** | Makes percentiles meaningful rather than indicative. |
| **Segment the corpus** | Startup NDAs and enterprise vendor NDAs have different norms. One pooled baseline hides that; comparing like with like would sharpen every finding. |
| **Measure extraction accuracy directly** | The eval currently measures the scorer end-to-end. A separate check of extraction against hand-labelled clauses would localise failures instead of leaving them ambiguous. |
| **Confidence on extracted values** | Let the model flag where it is unsure, and mark those findings as provisional rather than presenting all findings with equal certainty. |
| **Jurisdiction awareness** | Even a small lookup of "is this clause type enforceable here" would catch cases the statistics cannot. |
| **Clause-level redline diffing** | Show the suggested wording inline against the original rather than as separate text. |
| **OCR for scanned PDFs** | Deliberately cut for scope. A real user's NDA is often a scan. |
| **Other contract types** | The pipeline is type-agnostic; each new type needs a taxonomy and attribute set, not new architecture. |

---

## Layout

| Path | Contents |
|---|---|
| `src/legal_ai/config.py` | Paths and model IDs. Every path derives from `PROJECT_ROOT`. |
| `src/legal_ai/schemas.py` | Clause taxonomy and attribute schemas — the contract everything reads from. |
| `src/legal_ai/parsing.py` | PDF/DOCX/TXT → clauses. Regex, no API calls. |
| `src/legal_ai/extract.py` | Clauses → typed attributes. One structured-output call per document. |
| `src/legal_ai/profile.py` | Clause-level facts → one document-level row. |
| `src/legal_ai/corpus.py` | Market distributions over document profiles. |
| `src/legal_ai/scoring.py` | Deviation scoring. Deterministic, no LLM. |
| `src/legal_ai/negotiate.py` | Flagged terms → negotiating positions. The generative stage. |
| `src/legal_ai/evaluation.py` | Measures the scorer against ground truth. |
| `analyze.py` | Analyse one NDA. |
| `build_corpus.py` | Build market statistics from `data/corpus/`. |
| `evaluate.py` | Run the eval. |
| `app.py` | Streamlit UI. Presentation only — `src/` never imports Streamlit. |
| `data/corpus/` | Reference NDAs the comparison is built from. |
| `data/golden/` | Eval documents plus `expected.json` ground truth. |

Nothing in `src/` imports Streamlit, so the pipeline is testable without a
browser and a different front end — or an API layer — is a matter of calling the
same functions.

---

## Scope

**Built:** NDA parsing, attribute extraction, market comparison, deviation
scoring, negotiating positions, email drafting, evaluation harness, web UI.

**Deliberately not built:** document drafting, model fine-tuning,
jurisdiction-specific legal reasoning, multi-document-type support, OCR.

The exclusions are scope decisions, not oversights. Each was cut to keep a small
pipeline working end to end rather than leaving several half-finished.
