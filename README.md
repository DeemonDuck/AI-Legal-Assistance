# NDA Reviewer

### ▶ **[Live demo — ai-legal-assist.streamlit.app](https://ai-legal-assist.streamlit.app/)**

[![tests](https://github.com/DeemonDuck/AI-Legal-Assistance/actions/workflows/tests.yml/badge.svg)](https://github.com/DeemonDuck/AI-Legal-Assistance/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-live-FF4B4B)](https://ai-legal-assist.streamlit.app/)

**Someone sent you an NDA to sign. Is it normal, or is it trying to take
advantage of you?**

This tool reads the document, compares it against other NDAs, and tells you in
plain English which parts are unusual — and what to ask for instead.

> ⚖️ **This is information, not legal advice. It does not replace a lawyer.**
> It can tell you how your document compares to others. It cannot tell you
> whether to sign.

**Nothing to install to try it.** Open the demo above and upload
`data/golden/aggressive_nda_01.txt` from this repository — a document with eight
deliberately unusual terms. It returns seven high-risk findings instantly and
without a single API call, because the analysis for the three sample documents
ships with the repo.

---

### Where to start

| You are… | Read |
|---|---|
| Just want to see it work | **[The live demo](https://ai-legal-assist.streamlit.app/)** — no install, no key |
| Judging this against the brief | [What the brief asked for](#what-the-brief-asked-for) |
| Just want to know what this does | Keep reading — the next two sections |
| Want to run it yourself | [Try it](#try-it) |
| A developer evaluating the build | [How it works](#how-it-works) and the ▸ expandable sections |
| Wondering what it *can't* do | [What it can't do](#what-it-cant-do) — please read this one |

---

## What the brief asked for

> *Legal documents are hard to understand without professional help. Build a
> GenAI-powered solution that helps users understand, compare, and navigate
> legal documents — providing information and assistance, not replacing a
> lawyer. GenAI integration is a mandatory requirement.*

Every clause of that brief maps to something you can point at in the running
app:

| The brief | Where it is in this project |
|---|---|
| **Understand** | Every clause gets a one-sentence plain-English summary, and every finding is written for a non-lawyer — *"your obligation never expires"*, not *"perpetual survival of confidentiality obligations"*. There is a [glossary](#plain-english-glossary) for the eight terms that cannot be avoided. |
| **Compare** | This is the core of the tool, not a feature beside it. The document is reduced to typed facts and each one is scored against the distribution of the same fact across eight reference NDAs. Every claim is a number you can check. |
| **Navigate** | Each finding cites the clause it came from and shows the original text on demand, so a user is never told something about their document without being shown where. |
| **Information, not legal advice** | The disclaimer renders on every screen rather than once at the start. The tool reports how the document compares and what to ask for; it never says whether to sign. |
| **GenAI is mandatory** | Two generative stages. The model does the *reading* (messy legal prose → typed facts) and the *writing* (arguing both sides of each flagged term, drafting replacement wording and the email). The comparison in between is deliberately arithmetic — see [why](#the-five-steps). |

**Scope was narrowed on purpose: NDAs only.** One contract type carried end to
end, with an eval proving it works, beats four types half-built. The pipeline
itself is type-agnostic — a new contract type needs a new question set, not new
architecture.

---

## The problem

Imagine a small design studio is asked to sign an NDA before pitching to a
large client. It is four pages of dense legal text. They skim it, see nothing
obviously alarming, and sign.

Buried in paragraph 7 is a clause saying they cannot work with any competing
business for two years. In paragraph 3, the confidentiality obligation never
expires — not in five years, not ever. Neither sentence looks unusual. Both are
ordinary English words in an ordinary-looking paragraph.

**A lawyer would catch both in about ninety seconds.** Not because they are
cleverer, but because they have read hundreds of NDAs and know what normal looks
like. The problem is not that the language is incomprehensible — it is that most
people have no comparison to judge it against.

That comparison is what this tool provides.

## What it does

You upload an NDA. You get back something like this:

```
[HIGH RISK]  Confidentiality term                        (see: 3. Term)

   This document : your obligation never expires
   Other NDAs    : all 8 compared expire — the longest after 60 months
   Why it matters: An obligation with no end date binds you indefinitely.
                   You can never know when it is safe to use what you learned.
```

Three lines, and each does a specific job:

- **What your document says** — in plain language, not legal quotation
- **What other NDAs say** — so you can see *how* unusual it is, not just that it is
- **Why you should care** — the practical consequence for you

Then, for anything flagged, it writes you a **negotiating position**: the
strongest honest argument the *other side* would make, the case against it,
replacement wording to ask for, and what to settle for if they say no. It can
draft the email too.

### The one idea behind the whole thing

Every statement it makes is **checkable**.

Not *"this clause seems aggressive"* — that is an opinion with nothing behind
it, and you would have no way to argue with it. Instead: *"this runs 36 months
where every NDA we compared uses 12."* That is a fact you can verify, repeat,
and say out loud in a conversation.

---

## Try it

### The hosted app — nothing to install

**<https://ai-legal-assist.streamlit.app/>**

Upload any of the three sample documents from [`data/golden/`](data/golden/):

| Sample | What to expect |
|---|---|
| `aggressive_nda_01.txt` | Eight deliberately unusual terms. Seven high-risk findings. |
| `clean_nda_02.txt` | Deliberately ordinary. Should stay quiet — this is the one that proves it discriminates. |
| `mixed_nda_03.txt` | Mostly normal, three planted problems. The hardest of the three. |

All three are instant and cost nothing: their analysis is committed to the
repository, so the deployed app serves them without calling a model.

> Uploading **your own** document does call the API, on a free tier shared by
> everyone who opens the link — expect roughly a minute per request, and see
> [Privacy](#privacy-and-handling-of-uploads) for what happens to the file.

### Run the dashboard on your own machine

```bash
py -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

cp .env.example .env               # then paste your key into it
.venv/Scripts/python.exe -m streamlit run app.py
```

That opens at **http://localhost:8501**.

Upload `data/golden/aggressive_nda_01.txt` — a document with eight deliberately
unusual terms. It should report seven high-risk findings **instantly and without
making a single API call**, because the analysis for the three sample documents
ships with the repository.

You do not need to build anything first. The market baseline
(`data/corpus_index/corpus_stats.json`) is committed, so a fresh clone is a
working program. Rebuild it only if you change the corpus or the extraction
schema:

```bash
py build_corpus.py                 # ~21 API calls, ~21 minutes on a free tier
```

<details>
<summary><b>▸ Getting a key</b></summary>

The default provider is **Groq**, whose free tier is enough to run everything
here. Create a key at [console.groq.com](https://console.groq.com/keys) and put
it in `.env`:

```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your-key-here
```

Anthropic works too — set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`
instead. Nothing else changes; the provider is swapped in one place
(`legal_ai/llm.py`) and the pipeline never learns which one it is talking to.

**Free tiers have two separate limits**, and the second one surprises people:
tokens *per minute* and tokens *per day*. This project paces itself against both
rather than firing requests and retrying whatever gets rejected. On Groq's free
tier expect roughly one request per minute, so analysing a fresh twelve-clause
NDA takes about three minutes. The waiting messages are the rate limiter working,
not a hang.

</details>

<details>
<summary><b>▸ Putting it online (Streamlit Community Cloud, free)</b></summary>

This is how <https://ai-legal-assist.streamlit.app/> is deployed. To stand up
your own copy:

1. Push your fork to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and
   choose **New app**.
3. Point it at your repository, branch `main`, main file `app.py`. Set the Python
   version to 3.11 or newer under **Advanced settings**.
4. Paste your key into the **Secrets** box. This is TOML, not `.env` — quotes and
   spaces around `=` are required:

   ```toml
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "gsk_your-key-here"
   ```

5. Deploy. The first build takes a few minutes.

`app.py` copies those secrets into the environment at startup, because Streamlit
Cloud exposes them through `st.secrets` rather than as environment variables. The
copy lives in the front end so that nothing under `src/` depends on Streamlit.

**A deployed instance shares your API key with everyone who opens the URL.** The
three sample documents cost nothing because their analysis is cached in the
repository, but any other upload spends your allowance. Treat a public link
accordingly, and see [Privacy](#privacy-and-handling-of-uploads) for what a
hosted instance does with the documents people give it.

Streamlit Cloud also sleeps an app that has gone unvisited. The first request
after that wakes it and takes a few seconds; nothing is lost but the cached
extractions of anything uploaded since the last restart.

</details>

<details>
<summary><b>▸ Command-line usage</b></summary>

```bash
# Check how the document was split into clauses. No API call, costs nothing.
py analyze.py --parse-only path/to/nda.pdf

# Deviation report
py analyze.py path/to/nda.pdf

# ...plus negotiating positions (1 extra API call)
py analyze.py --negotiate path/to/nda.pdf

# ...plus a draft negotiation email
py analyze.py --email path/to/nda.pdf

# Corpus and evaluation
py build_corpus.py --show     # print saved market statistics, no API calls
py evaluate.py                # measure the scorer against ground truth

# What changed between two extraction schema versions. No API calls.
py compare_runs.py --doc aggressive_nda_01

# Offline tests — no API key, no network, no cost
py tests/run_all.py                   # all eight suites
py tests/run_all.py --verbose         # with each suite's full output
py tests/test_scoring.py              # or one at a time, which is how you debug

# Lint
ruff check .
```

Results are cached by document content, so re-running a document is instant and
free. Delete `.cache/` to force re-extraction.

`evaluate.py` refuses to run if the market baseline was built under a different
extraction schema than the one in the code. Comparing documents read by one set
of field descriptions against a baseline built from another produces a plausible
score rather than an error, so it is stopped rather than warned about. Rebuild
with `py build_corpus.py`, or pass `--allow-stale-corpus` to reproduce an earlier
measurement deliberately.

Moving between machines: see [DEVICE_SETUP.md](DEVICE_SETUP.md). Use git, not a
zip — `.venv` contains absolute paths and breaks when moved.

</details>

---

## How it works

**In one sentence:** it turns a legal document into a spreadsheet of facts, then
checks which of those facts are unusual.

The trick is that you cannot compare two contracts by comparing their *words* —
contracts say the same thing in a hundred different ways. But you *can* compare
them by their **numbers and yes/no answers**:

| Question about the document | This NDA | Typical NDA |
|---|---|---|
| How many months does confidentiality last? | never ends | 24–36 |
| Do the obligations run both ways? | No | Usually yes |
| Are the standard protections included? | None of them | All five |
| Is there a non-compete? | Yes, 24 months | Never |

Once a contract is reduced to a table like that, spotting the odd one out is
straightforward arithmetic — and arithmetic can be shown to the user and checked.

### The five steps

```
Your NDA (PDF / Word / text)
    │
    ├─ 1. Split into clauses                     pattern matching, no AI
    │
    ├─ 2. Read each clause, pull out the facts   ← AI does the reading
    │     "how long", "both ways?", "what's missing"
    │
    ├─ 3. Combine into one summary of the document
    │
    ├─ 4. Compare against the reference NDAs     plain arithmetic
    │     flag anything unusual in a way that hurts you
    │
    └─ 5. Argue both sides of each flag          ← AI does the writing
          risk level, explanation, suggested wording, fallback
```

**AI is used at steps 2 and 5, and deliberately not at step 4.** Reading messy
legal prose and writing persuasive arguments are things AI is genuinely good at.
Deciding whether 36 is unusual compared to 12 is arithmetic — and arithmetic
gives the same answer every time, which matters when someone asks why the report
changed.

<details>
<summary><b>▸ Technical architecture</b></summary>

| Stage | Implementation | Rationale |
|---|---|---|
| Parsing | Regex heading detection with paragraph fallback | Runs on every document; cheap to get approximately right, and extraction recovers from slightly wrong boundaries |
| Extraction | One structured-output call per document, Pydantic-validated | Clauses are interdependent — mutuality, whether carve-outs appear anywhere, cross-references. Per-clause calls are blind to that and multiply latency ~12× |
| Profiling | Pure logic | Collapses clause-level facts into one document-level row |
| Scoring | Pure arithmetic | Deterministic and falsifiable |
| Negotiation | One structured-output call | Both stances plus synthesis in a single response |

Data flow:

```
list[RawClause]        (index, heading, text)
      ↓ extract.py — cached by sha256(clauses + model + schema version)
list[ExtractedClause]  (clause_type, party_favored, 16 typed attributes)
      ↓ profile.py — aggregation rules per attribute
DocumentProfile        (one comparable row)
      ↓ corpus.py — same pipeline over data/corpus/
CorpusStats            (n, median, p25/p75/p90, true-fractions, raw values)
      ↓ scoring.py
DeviationReport        (ranked findings with clause citations)
      ↓ negotiate.py
NegotiationSet         (positions, redlines, fallbacks)
```

Model is configurable in `src/legal_ai/config.py`. Extraction and negotiation
have separate model constants so the high-volume pass can be downgraded
independently.

</details>

---

## Why it is built this way

### We do not compare documents by similarity

The obvious approach — and the one originally planned — was to measure how
*similar* each clause is to a typical clause, and flag the ones that look
different.

**It does not work, and the reason is interesting.** Similarity measures whether
two pieces of text are *about the same thing*. It does not measure whether the
*terms are worse*.

"Confidentiality lasts two years" and "confidentiality lasts ten years" are
almost identical sentences. One number differs. A similarity score sees two
nearly-identical clauses and says "normal" — while the second one binds you for
five times as long.

Worse, it fails in the opposite direction too: a perfectly ordinary clause
written in unusual style looks *very* different, and gets flagged as dangerous
when it is fine.

So the tool would flag **odd writing** and ignore **bad terms** — exactly
backwards. And it would look convincing, because the output would still be full
of numbers.

<details>
<summary><b>▸ The technical version</b></summary>

Embedding cosine similarity measures topical sameness, not deal-term deviation:

| Comparison | Cosine similarity | Materially different? |
|---|---|---|
| "term of two (2) years" vs "term of ten (10) years" | ~0.95 | Enormously — 5× the obligation |
| A mutual clause vs its one-way version | ~0.93 | Yes — completely different risk |
| A standard clause in unusual house style | ~0.70 | No — same terms, different words |

Both error modes follow: false negatives on predatory terms that embed close to
standard ones, and false positives on stylistic variation. Crucially the output
still *looks* numeric and plausible, so without an eval set the failure is
invisible.

The fix splits two conflated jobs. Clause-type classification is genuinely a
retrieval problem, and the LLM does it free in the pass it already makes.
Deviation detection becomes distribution comparison over typed attributes.

**What it bought:** a checkable claim instead of an uninterpretable distance; a
concrete handoff to the negotiation stage; and four heavy dependencies removed
(`torch` ~2GB, `sentence-transformers`, FAISS, vector store), so a cold clone
installs in about a minute.

**The honest cost:** semantic retrieval is lost. A clause worded in a way the
taxonomy does not anticipate lands in `OTHER` and gets weaker treatment. NDA
clauses are highly conventional so this is small in practice — but it is real.

</details>

### Being unusual is only bad in one direction

A two-week confidentiality term is just as unusual as a fifty-year one. Only one
of them is a problem for you.

So every fact the tool checks knows which direction is bad. Unusually *good*
terms are reported too — separately, as negotiating leverage rather than
warnings.

Some invert in ways that are easy to get wrong: a **shorter** deadline to return
documents is *worse* for you, not better, because it is harder to comply with.

### The other side gets a fair hearing

For each flagged term the tool states the strongest honest argument for why the
other party wants it — before arguing against it.

This is not politeness. **An argument that cannot explain the other side's
reasoning is a complaint, not a negotiating position**, and it will lose the
moment someone pushes back. Knowing why they want something is how you find what
they will accept instead.

<details>
<summary><b>▸ Why one call and not a multi-turn debate</b></summary>

The original plan specified two persona agents debating with a third
synthesising — three sequential calls per flagged clause. On a document with
eight flags that is 24 calls and 30–60 seconds of latency, live. Multi-turn
debate also tends toward mush, since each round regresses toward the mean of the
two positions.

The schema instead forces both stances and the synthesis into one structured
response, with field ordering that requires the counterparty justification
before the rebuttal. Same analytical content, a third of the cost and latency.

Each term is passed its specific corpus comparison *and* the original clause
text. The comparison must be cited in the argument; the clause text is needed
because a redline drafted without the original reads like it came from a
different agreement.

</details>

### Two opinions that are allowed to disagree

The arithmetic says how **unusual** a term is. The AI says how **dangerous** it
is. These are different questions — a rare term can be harmless, a common one
can be serious.

When they disagree, the tool shows you both rather than hiding the discrepancy.

---

## How we know it works

A tool like this fails in the worst possible way: it produces a clean,
confident, well-formatted report that happens to be wrong. Nothing looks broken.

So it is tested against documents where **the right answer was written down in
advance**:

| Test document | What it checks |
|---|---|
| One with eight deliberately bad terms | Does it find them? |
| One that is deliberately ordinary | Does it stay quiet? |
| One mostly normal with three bad terms | Can it tell the difference? |

The third matters most. **A tool that flags everything catches every problem and
is completely useless.** The ordinary document is what stops that.

Currently **11 of 11 planted terms found, with no false positives**: eight of
eight on the aggressive document, three of three on the mixed one, and silence
on the control. The mixed document raised one high-risk finding against a budget
of two — it discriminates rather than flagging in bulk.

Read that number with the caveat the tool prints for itself: it measures the
scorer against **eight reference NDAs**, which is not the same as measuring it
against the market. Run `py evaluate.py` to reproduce it.

### Underneath that, eight offline suites

`py evaluate.py` answers *"are the answers right?"*. It needs a corpus baseline
and costs API calls. Beneath it sit eight suites that answer *"is the machinery
correct?"* on fabricated data, with no key, no network and no cost:

| Suite | What it pins down |
|---|---|
| `test_parsing.py` | Clause splitting and loading. The stage every later stage trusts blindly — a badly-cut clause still extracts, still scores, and still prints a confident report. |
| `test_stats.py` | Aggregation rules and percentile maths. A bad percentile produces a plausible number rather than an error. |
| `test_scoring.py` | Deviation scoring against a synthetic market, including the inverted-direction rules. |
| `test_evaluation.py` | **The eval harness itself** — fed deliberately broken scorer output and required to report failure. |
| `test_negotiate.py` | Cache keys, prompt grounding, and the no-risk short circuit that avoids spending a request. |
| `test_accessibility.py` | WCAG contrast ratios recomputed from the shipped palette and theme. |
| `test_session_state.py` | That one document's generated positions and draft email cannot survive into the next upload. |
| `test_llm.py` | The provider seam: strict-schema rewriting, token estimation that must never under-reserve, and rate-limit pacing driven by a fake clock. |

[CI](.github/workflows/tests.yml) runs all eight on Python 3.11 and 3.13, lints
with ruff, and separately checks the two things that fail *silently* in
production: that the committed corpus baseline still matches the extraction
schema, and that the three demo documents still hit their cached extractions —
if they stop, the public demo quietly starts spending the day's token allowance
on every visitor.

<details>
<summary><b>▸ Methodology</b></summary>

Two rules keep the measurement honest:

1. **Ground truth was written by reading the documents**, never by recording what
   the scorer produced. Recording current output as "expected" makes the eval a
   tautology that passes however wrong the scorer is, and locks today's bugs in
   as tomorrow's requirements.
2. **Eval documents are not in the corpus.** Scoring a document against a
   baseline containing it leaks the answer — its own values pull the median
   toward itself, so it scores as more normal than it is. `check_leakage()`
   enforces this on every run rather than trusting it.

Measured: recall against planted terms, false positives against a declared HIGH
budget, and a severity floor — detecting a perpetual term but rating it LOW is
a failure, because severity is what decides whether a user acts.

The eval also tests itself. `tests/test_evaluation.py` feeds the harness
deliberately broken scorer output and requires it to report failure: an empty
report must not pass, a partial detection must not pass, and a flag-everything
scorer must not pass on recall alone. **An eval that cannot fail is worse than
no eval** — it produces a number that looks like evidence while proving nothing.

</details>

---

## Privacy and handling of uploads

An NDA is usually a draft agreement someone has not signed yet, so it is worth
being exact about where it goes.

| | |
|---|---|
| **The uploaded file** | Written to a temp file because the parser reads from disk, and deleted as soon as the text has been read — in a `finally`, so it goes even when extraction fails. It is never written into the repository or any persistent store. |
| **The clause text** | Sent to the configured model provider (Groq by default, Anthropic optionally) for the extraction and negotiation calls. This is the one thing that leaves the process, and it is inherent to the tool rather than incidental. |
| **The extraction result** | Cached on the server, keyed by a hash of the clause text. This is what makes re-running a document instant and free. On the hosted demo that cache lives on the container's own disk and disappears when it restarts — but until then, a document's *extracted facts and summaries* do persist there. |
| **Analytics** | Off. `gatherUsageStats = false` in `.streamlit/config.toml`, because uploads here are draft legal agreements and telemetry should not be an exception to that. |
| **Upload size** | Capped at 10 MB at the file picker rather than after the upload, since the parser would reject a large scan anyway. |

**The hosted demo is a demo.** It runs on one shared free-tier API key, and the
sensible thing to do with a genuinely confidential agreement is to run the app
locally, where the key and the cache are yours.

Secrets never reach the repository: `.env` and `.streamlit/secrets.toml` are
both gitignored, and `config.py` checks the key's prefix so a key pasted into
the wrong provider's variable fails with an explanation instead of a bare 401.

---

## What it can't do

Please read this section. A tool making claims about legal documents should be
clear about its limits.

**It is not a lawyer and it cannot tell you whether to sign.** It tells you how
your document compares to others. Deciding what to do with that is yours.

**The comparison set is small and artificial.** The eight reference NDAs are
*written examples*, not real agreements collected from the wild. They are
realistic, but nobody should read "unusual compared to these eight" as "unusual
in the real world". The tool words its own output carefully for this reason, and
says so on every report.

**Eight documents is not enough for confident statistics.** Fifteen to twenty is
the realistic minimum. The tool warns you about this itself.

**It does not know the law in your country or state.** A two-year non-compete is
unenforceable in California and routine in Texas. This tool notices *that* your
NDA has one, not whether a court near you would uphold it.

**It can misread a clause.** The AI reading step is good but not perfect, and a
misread produces a confident, well-formatted, wrong finding. Every report cites
the clause it came from — if something looks wrong, check the original text.

**NDAs only.** Other contracts will load, but the questions it asks are
NDA-specific and most clauses will come back unclassified.

**No scanned documents.** It needs real text, not a photograph of text. A scanned
PDF is rejected with an explanation rather than analysed badly.

---

## What would make it better

Roughly ordered by improvement per unit of work:

| Improvement | Why it matters |
|---|---|
| **Use real NDAs as the comparison set** | The single highest-value change. Public company filings contain thousands of real ones. Turns every claim from "unusual compared to our examples" into a defensible statement about the market. |
| **More of them — 20, 50, 100** | Makes the statistics meaningful rather than indicative. |
| **Count silence as data, not absence** | Some findings currently rest on one or two reference documents, because a document that says nothing about indemnity contributes nothing to that attribute's baseline. But silence *is* the answer — an NDA that never mentions indemnity does not impose one. Counting those documents would take several denominators from 1 or 2 up to 8, and would change some severities. |
| **Separate comparison sets by context** | Startup NDAs and enterprise vendor NDAs have genuinely different norms. One pooled baseline hides that; comparing like with like would sharpen every finding. |
| **Test the reading step on its own** | The eval currently measures the whole pipeline. Checking extraction against hand-labelled clauses would localise failures rather than leaving them ambiguous. |
| **Let the AI say when it is unsure** | Mark uncertain readings as provisional instead of presenting every finding with equal confidence. |
| **Basic jurisdiction awareness** | Even a small "is this enforceable here" lookup would catch what statistics cannot. |
| **Show redlines inline** | Suggested wording marked up against the original, rather than as separate text. |
| **Support scanned documents** | Cut for scope, but a real user's NDA is often a scan. |
| **Other contract types** | The pipeline is type-agnostic. Each new type needs a question set, not new architecture. |

---

## Plain-English glossary

| Term | What it means |
|---|---|
| **NDA** | Non-Disclosure Agreement. A contract promising to keep information secret. |
| **Mutual / one-way** | Mutual means both sides must keep secrets. One-way means only you must. |
| **Carve-out** | An exception. E.g. you are not bound to keep secret something that was already public. Their absence is a common trap. |
| **Non-compete** | A promise not to work with competing businesses. Unusual in an NDA — it restricts your livelihood, not just your speech. |
| **Non-solicit** | A promise not to hire the other side's employees. |
| **Indemnity** | A promise to cover the other side's losses, which can far exceed anything you gain. |
| **Perpetual** | Never expires. |
| **Redline** | A proposed edit to contract wording. |

---

<details>
<summary><b>▸ For developers: project layout</b></summary>

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
| `app.py` | Streamlit UI. Presentation only. |
| `compare_runs.py` | Diff two extraction schema versions over the same document. |
| `data/corpus/` | Reference NDAs the comparison is built from. |
| `data/golden/` | Eval documents plus `expected.json` ground truth. |
| `tests/run_all.py` | Runs every offline suite; non-zero exit on any failure. |
| `pyproject.toml` | Ruff configuration. No `[project]` table on purpose — see the comment at the top of the file. |
| `.github/workflows/` | CI: lint, the eight suites on two Python versions, and the two production checks that fail silently otherwise. |

**Nothing in `src/` imports Streamlit.** The pipeline is testable without a
browser, and adding an API layer or a different front end means calling the same
functions rather than untangling them. This is also what made dropping the
originally-planned FastAPI backend a deferral rather than a loss.

Two things not to "tidy up":

- **Schema fields are required-but-nullable** (`X | None` with no default), not
  optional-with-defaults. Strict structured output requires every field in
  `required` with `additionalProperties: false`; nullable-required is how the
  model says "this clause states no term" without violating the schema.
- **`SCHEMA_VERSION` is part of the cache key.** Bump it when schemas change
  shape, or stale results from the old schema get served as current.

</details>

---

## Scope

**Built:** NDA parsing, attribute extraction, market comparison, deviation
scoring, negotiating positions, email drafting, evaluation harness, web UI.

**Deliberately not built:** document drafting, model fine-tuning,
jurisdiction-specific legal reasoning, multi-document-type support, OCR.

These are scope decisions, not oversights. Each was cut to keep one small
pipeline working end to end rather than leaving several half-finished.
