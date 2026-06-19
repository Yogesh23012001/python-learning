# Contextual Retrieval (Anthropic)

## The problem (my words)

Chunking destroys context. To retrieve precisely I want small, self-contained
chunks — but the act of cutting a chunk small enough to retrieve cleanly strips
away the document context that made it *mean* something. A chunk can end up
saying "the revenue grew 3%" with no clue *which company* or *which quarter* —
the surrounding document had that, the isolated chunk doesn't (the classic ACME
revenue example).

I saw this exact failure in my own corpus on Monday: the 60-char starved chunk
from chunking, and the citations/code-snippet chunks that start mid-thought.
When I ran the fragment query Wednesday it surfaced real ones — e.g. a chunk
that literally begins `". In internal evaluations, adaptive thinking..."` with a
bare leading period and no subject. Embedded alone, that chunk has almost no
retrievable meaning, even though the information is fine in context.

## The mechanism (my words)

For each chunk: send the LLM the **whole document plus that chunk**, and ask it
to write a short snippet that situates the chunk within the document. Prepend
that snippet to the chunk, then embed the *contextualized* chunk. The context is
**informed by the whole document but specific to this chunk** — so the chunk now
carries the situating detail it lost when it was cut. Because the prepended text
is just words, this helps **both** dense (vector) retrieval *and* BM25 (the added
terms give the keyword index something to match too).

## The numbers

Anthropic's published results (retrieval-failure reduction vs. a plain baseline):

- **Contextual embeddings alone:** ~35% fewer failures.
- **Contextual embeddings + contextual BM25:** ~49% fewer.
- **+ reranking on top:** up to ~67% fewer.

Real, significant gains — and they *stack* with the hybrid + reranking machinery
I already built. Contextual retrieval fixes the chunk *before* embedding;
hybrid/rerank improve what happens *after*. Different stages, additive wins.

## The cost trick (my words — this one matters to me)

One LLM call per chunk sounds expensive — naively it is. **Prompt caching** is
what makes it affordable: the whole document is sent alongside *every* chunk of
that document, so if I cache the document, the first chunk pays full price to
write the cache and every chunk after it reads the document at ~10% of normal
cost. Anthropic quotes **~$1.02 per million document chunks** as a one-time
ingestion cost.

This is the same prompt-caching feature that lives in my own corpus work — the
technique is only viable *because of* caching. That's the through-line of this
whole month: cost-governance isn't a separate concern bolted on afterward, it's
the thing that makes the mechanism possible at all. Same instinct as metering
in the gateway.

## My fixable chunk

Real context-starved chunks from my corpus, surfaced by the fragment query
(`length < 200 OR LIKE '],%' OR LIKE '%},%'`). Two distinct failure modes — and
one chunk that contextual retrieval *can't* save. The **Before** text is the
real chunk as embedded today; I'll fill in the **After** (the contextualized
version) once I run the contextualize + re-embed test.

### Fixable #1 — prose that starts mid-sentence

`[prompt-engineering/claude-prompting-best-practices]`

**Before** (as embedded today):

> `. In internal evaluations, adaptive thinking reliably drives better
> performance than extended thinking. Consider moving to adaptive thinking to
> get th...`

Bare leading period, no subject. Nothing tells the embedder this is about
Claude's thinking modes. A query like "adaptive vs extended thinking?" has
little to lock onto.

**After:** _(to fill in after the contextualize test)_

### Fixable #2 — a bare code fragment

`[build-with-claude/prompt-caching]`

**Before:**

> `max_tokens=1024,\n    cache_control={"type": "ephemeral"},\n    system="You
> are a helpful assistant that remembers our conversation.",\n    messages=[\n`

A slice of a `messages.create(...)` call with no function name, no surrounding
prose. Embedded alone it's nearly meaningless — the words `cache_control` and
`ephemeral` are the only signal, and a natural-language query won't match raw
kwargs well.

**After:** _(to fill in after the contextualize test)_

### Not fixable — boilerplate, drop it at ingestion

`[tool-use/overview]`

> `Browse all tools\nDirectory of Anthropic-provided tools and properties.\nWas
> this page helpful?`

This one is short and starved too, but contextualizing it is pointless — it's
nav/footer boilerplate ("Was this page helpful?"), not real content. Prepending
context to junk just produces well-labeled junk. The right fix here is the
**loader's `_strip_boilerplate` / length gate** — drop it at ingestion, don't
contextualize it. That's the dividing line: contextual retrieval rescues
*starved-but-real* chunks; boilerplate stripping removes *not-real* ones. Two
different tools for two different problems.


## Contextual retrieval — carry-forward for Saturday capstone (Wed H4)
Built + working: generates good doc-aware context (the `effort` parameter
appeared from elsewhere in the doc — whole-doc context proven). 5th mode wired.
Caching trick implemented (doc block cached, stable-prefix-first).

TWO ISSUES to fix in the production capstone:
1. ONE-PASS INGESTION. Contextualizing in a SECOND run matched chunks to stored
   rows by (doc_id, chunk_text) — fragile: live docs shifted between runs, 91 of
   677 chunks failed the exact-text match and silently got `continue`d. They have
   contextual_embedding IS NULL. Fix: chunk → contextualize → embed BOTH versions
   → store, all in ONE pass. No cross-run text matching.
2. CONCURRENCY. 677 sequential API calls took 22.5 min (~2s/chunk). Caching cut
   COST not LATENCY. Fix: fire contextualization calls concurrently (asyncio),
   ~22 min → a few.
For Thursday eval: 586/677 chunks contextualized — enough to measure the effect,
but note the gap so the contextual-mode numbers account for it.

