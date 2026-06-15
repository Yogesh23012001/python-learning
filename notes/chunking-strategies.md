# Chunking Strategies

## Why chunking is the ceiling (my words)

Chunking happens **before** the embedder ever runs, so it caps everything
downstream. Three failure modes:

- **Split fact** — a single fact gets cut across a boundary, so *neither* chunk
  contains the whole answer. My fixed-size run did this literally: chunk 1 ends
  "...models g" and chunk 2 starts "ws nothing after its training date" — the
  word **"knows" was sliced into "kno|ws"** mid-word. No retrieval can rejoin
  what was severed at ingest.
- **Diluted chunk** — too large, so one chunk covers several topics; its
  embedding is an average of all of them and points sharply at none. It ranks
  mediocre for every query.
- **Starved chunk** — too small, so the chunk loses the context that gives it
  meaning. My recursive run produced a **60-char chunk** (min=60) — a fragment
  that, embedded alone, barely says anything.

The key realization: **no embedder, index, or reranker can fix a bad chunk.**
They all operate on whatever the chunker handed them. Garbage boundaries in →
garbage retrieval out. The tension underneath it all: **smaller = focused
meaning but lost context; larger = self-contained but diluted meaning.**

## The four (+1) strategies (my words)

- **Fixed-size** — cut every N characters, ignore the text. Fast and dumb;
  splits mid-word/mid-sentence (see "kno|ws" above). 6 chunks here.
- **Fixed-size + overlap** — same cut, but each chunk repeats the last ~50
  chars of the previous one, so a fact split at a boundary survives *whole* in
  one of the two overlapping chunks. Costs duplication. 7 chunks.
- **Sentence-based** — split on sentence boundaries (`. `). Natural chunks that
  read cleanly, but sizes vary and a very long sentence still overflows.
  8 chunks.
- **Recursive** — try the biggest natural break first (paragraph), fall back to
  sentence, then word. Respects document **structure**, so chunks break where
  the document already breaks. The usual best default. 8 chunks.
- **(+1) Semantic** — split where the *meaning* shifts (embed candidate
  windows, cut at similarity drops). Most faithful to topic boundaries, most
  expensive. Didn't benchmark it today — noting it as the next rung.

## What the comparison showed

| strategy            | chunks | avg | min | max |
|---------------------|--------|-----|-----|-----|
| Fixed-size (300)    |   6    | 278 | 165 | 300 |
| Fixed + 50 overlap  |   7    | 281 | 165 | 300 |
| Sentence-based      |   8    | 206 | 120 | 266 |
| Recursive           |   8    | 213 |  60 | 296 |

What the numbers and the actual text revealed:

- **Fixed-size split mid-word.** "knows" became "kno" + "ws" across chunks 1→2.
  Most uniform sizes (tight 165–300 band) precisely *because* it ignores
  meaning — uniformity bought at the cost of broken facts.
- **Overlap duplicated text to heal the boundary.** With 50-char overlap the
  second chunk starts "source text. Second, knowledge cutoff..." — the trailing
  "...retrieved source text" from the prior chunk is repeated, so the fact
  spanning the boundary now appears intact in one chunk. One extra chunk (6→7)
  is the duplication cost.
- **Sentence-based started clean.** Chunk 2 begins exactly at "Second,
  knowledge cutoff:" — a real sentence boundary, no severed words. Sizes spread
  wider (120–266) as expected.
- **Recursive aligned with structure but produced a starved chunk.** It honored
  paragraph/sentence breaks, but also emitted that 60-char fragment — proof that
  "respect boundaries" and "good size" are *separate* knobs.

## My pick for the RAG corpus

**Recursive, with overlap.** It respects document structure (breaks where the
doc already breaks) and the overlap buys back context at boundaries so a fact
split between two chunks survives whole in one of them — directly defusing the
"split fact" failure mode I watched happen.

Starting point: **chunk_size ≈ 500 chars (~120 tokens), overlap ≈ 50–75 chars.**
The 300-char runs above ran a bit small (and recursive bottomed out at 60), so
I'd size up to keep chunks self-contained without diluting.

But the real point: **chunk size is a hyperparameter I measure, not guess.**
These are starting values — I'll tune chunk_size and overlap against retrieval
metrics (P@1/MRR) in **Thursday's eval**, the same way I'd tune any other knob.

## The thread to Wednesday

That 60-char starved chunk is exactly the problem **Contextual Retrieval
(Anthropic)** targets: before embedding, it prepends a short, document-level
description of *what this chunk is about* back into the chunk, so a fragment
that lost its context at split time gets that context restored prior to
embedding. Today's chunking sets up that fix — I now have a concrete starved
chunk to point at when I wire up contextual retrieval Wednesday.
