# Market corpus

Every comparison this project makes is grounded in these documents. They are the
evidentiary basis for statements like *"85% of market NDAs are mutual"*, so their
provenance matters more than any other data in the repo.

## What is currently here

**Eight synthetic reference drafts, not sampled real-world agreements.**

They were written to span the dimensions the scorer compares — confidentiality
terms from 18 to 60 months, mutual and one-way, complete and partial carve-out
sets, five jurisdictions, courts and arbitration, with and without
non-solicitation — so the pipeline has a realistic distribution to work against.

They are *representative in structure*, but they are **not evidence of what the
market actually does**. Nobody sampled real NDAs to produce them.

## What this means for claims

Until real templates replace these, the honest phrasing is:

> "compared against our reference corpus of 8 NDAs"

and **not**:

> "85% of market NDAs do X"

`CorpusStats.credibility_note()` generates this warning automatically and the
report surfaces it. Do not route around it — an unfalsifiable statistic
presented as fact is the fastest way to lose credibility with anyone who knows
the domain.

## Replacing these with real NDAs

The pipeline does not care where documents come from. Drop `.txt`, `.docx`, or
text-layer `.pdf` files into this directory and run:

```bash
py build_corpus.py --provenance "describe the real source here"
```

Keep `--provenance` accurate; it is stored in the stats file and shown to users.

Aim for **15+ documents** — below that, quartiles are noise rather than signal,
and `build_corpus.py` will tell you so.

Good sources of genuinely public NDAs:

- **SEC EDGAR full-text search** — confidentiality agreements filed as exhibits
  to public filings. Real, current, and public record.
- **University and government technology-transfer offices** — many publish their
  standard NDA templates.
- **Open legal template projects** — check the licence before redistributing.

One caution: a corpus drawn entirely from one side of the market (say, only
large-company templates) encodes that side's norms as "standard". A mix of
company sizes and both mutual and one-way agreements gives a more honest
baseline.
