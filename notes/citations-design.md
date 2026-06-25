# Citations Design

## Why citations (my words)

Citations turn a RAG answer from something you *trust* into something you can
*audit*. Three reasons they earn their place:

- **Verifiability / trust** — the user (or I) can follow a citation to the exact
  source and confirm the claim, instead of taking the model's word for it.
- **Hallucination defense** — a sentence with no source to point at is a visible
  red flag. The *absence* of a citation is itself the signal.
- **Pipeline debugging** — when an answer is wrong, the citation tells me *which
  chunk* misled the model, which separates a **retrieval** failure (wrong chunk
  fetched) from a **generation** failure (right chunk, bad synthesis). Without
  citations I can't tell those two apart.

The deeper shift is from "trust the answer" to "audit the answer." And it ties
straight back to faithfulness from my eval work: faithfulness measures
groundedness for *me*, the evaluator, after the fact; citations expose that same
groundedness to the *user*, inline, at read time. Same property — groundedness —
surfaced to a different audience.

## The two approaches (my words)
- Model-generated (inline): model cites [1][2] as it writes. Natural, granular,
  but the citation can hallucinate. Anthropic's Citations API fixes this by
  constraining cites to exact document spans.
- Structural (post-hoc): decompose answer into claims, verify each against
  chunks programmatically. Rigorous, catches ungrounded claims, but costs the
  verification step (same as my eval harness's faithfulness check).

## My choice + why

**Model-generated inline citations** for this pipeline. The model writes [1],
[2] as it answers, and I track the retrieved chunks so each [n] resolves to a
real chunk with its `doc_id` and `url`. Why this over the alternatives:

- **Cheap + natural** — no extra verification pass; the citations fall out of the
  same generation call that produces the answer.
- **Enough to debug** — every [n] maps to a chunk I actually retrieved, so I get
  the audit trail (which claim ← which chunk) for free.

The tradeoff I'm explicitly accepting: **the model self-reports which chunk it
used.** I trust that mapping but don't verify it per-claim — the model could cite
[2] while really paraphrasing [3], or cite a chunk that doesn't fully support the
sentence. Production-grade would close that gap one of two ways: Anthropic's
**Citations API** (cites constrained to exact document spans, so the citation
*can't* hallucinate) or a **post-hoc verification** step (the same claim-by-claim
faithfulness check my eval harness already runs). I'm choosing the cheap,
unverified version on purpose — and writing down exactly where it would need
hardening.

## The data model

A cited answer is the answer text with [n] markers, plus a citations list
mapping each marker back to its source chunk. My `RAGAnswer` already carries
`retrieved` (the chunks), so citations are really just the **answer→chunk
mapping** layered on top of what I have.

```
CitedAnswer:
  answer: str            # with [1], [2] inline markers
  citations: list[Citation]
Citation:
  marker: int            # the [n]
  doc_id: str
  url: str
  snippet: str           # the chunk text cited
```

`doc_id` and `url` come straight from the loader (`doc_id` is the page slug,
`url` the source page); `snippet` is the cited chunk's text. So a [n] is a
clickable, verifiable pointer — not just a number floating in the prose.

## The groundedness check

A claim *with* a citation = grounded (traceable to a source). A claim with *no*
citation = either general knowledge (which my grounding prompt forbids — it must
answer only from the retrieved context) or an ungrounded claim, i.e. a
**faithfulness violation**.

So **citation coverage is itself a groundedness signal**: what fraction of the
answer's claims carry a citation? 100% coverage means every claim at least
*purports* to be grounded; anything less flags specific sentences to audit. It's
a cheap proxy for faithfulness — not as rigorous as my eval harness's per-claim
verification (coverage *trusts* the marker; faithfulness *checks* it), but it's
computable from the answer alone, with zero extra judge calls. Coverage catches
*missing* grounding; the faithfulness pass catches *wrong* grounding. Two
cheap-vs-rigorous ends of the same groundedness question.
