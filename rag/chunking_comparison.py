"""Compare four chunking strategies on the same document.

For each: number of chunks, avg/min/max chunk size, and a peek at how it splits.
The goal is to SEE that the same document carves very differently, and to feel
the small-vs-large, focused-vs-self-contained tension.

Run: uv run python rag/chunking_comparison.py
"""

from __future__ import annotations

import statistics

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# A sample document — a few paragraphs with structure, like real docs.
DOCUMENT = """\
Retrieval-Augmented Generation (RAG) addresses three core limitations of language models. \
First, hallucination: models generate plausible but incorrect facts when asked about things \
outside their confident knowledge. RAG grounds answers in retrieved source text. Second, \
knowledge cutoff: a model knows nothing after its training date. With RAG, the knowledge lives \
in the corpus, not the weights. Third, private data: models were never trained on your internal \
documents. RAG lets a model answer using documents it has never seen.

The naive RAG pipeline has two phases. Ingestion happens once: documents are chunked, each chunk \
is embedded into a vector, and the vectors are stored in a vector database. Querying happens per \
request: the question is embedded with the same model, the top-k most similar chunks are retrieved, \
and those chunks are passed to the LLM which answers using them.

Chunking strategy matters more than most people expect. If a chunk is too large, its embedding \
becomes an average of several topics and points nowhere specific. If a chunk is too small, it loses \
the context needed to be meaningful on its own. The art is splitting text so each chunk is focused \
enough to have clear meaning yet self-contained enough to stand alone. A fact split across a chunk \
boundary can never be retrieved whole, no matter how good the embedding model is.

Hybrid search combines dense vector search with sparse keyword search like BM25. Pure vector search \
can miss rare keywords, acronyms, and exact identifiers. A query for "RFC 7519" might semantically \
drift, while keyword search nails the exact token. Combining both captures meaning and exactness.
"""


def stats(chunks: list[str]) -> dict:
    sizes = [len(c) for c in chunks]
    return {
        "n": len(chunks),
        "avg": round(statistics.mean(sizes)),
        "min": min(sizes),
        "max": max(sizes),
    }


def show(label: str, chunks: list[str]) -> None:
    s = stats(chunks)
    print(f"\n=== {label} ===")
    print(f"  chunks={s['n']}  avg={s['avg']} chars  min={s['min']}  max={s['max']}")
    print(f"  first chunk: {chunks[0][:120]!r}")
    if len(chunks) > 1:
        print(f"  second chunk: {chunks[1][:120]!r}")


# ============================================================
# Strategy 1 — Fixed-size (character count, no respect for boundaries)
# ============================================================
fixed = CharacterTextSplitter(
    separator="",  # pure character split, ignores structure
    chunk_size=300,
    chunk_overlap=0,
)
show("1. Fixed-size (300 chars, no overlap)", fixed.split_text(DOCUMENT))


# ============================================================
# Strategy 2 — Fixed-size WITH overlap (mitigates the split-fact problem)
# ============================================================
fixed_overlap = CharacterTextSplitter(
    separator="",
    chunk_size=300,
    chunk_overlap=50,  # each chunk repeats the last 50 chars of the previous
)
show("2. Fixed-size with 50-char overlap", fixed_overlap.split_text(DOCUMENT))


# ============================================================
# Strategy 3 — Sentence-based (split on sentence boundaries)
# ============================================================
sentence = CharacterTextSplitter(
    separator=". ",  # split on sentence ends
    chunk_size=300,
    chunk_overlap=0,
)
show("3. Sentence-based (split on '. ')", sentence.split_text(DOCUMENT))


# ============================================================
# Strategy 4 — Recursive (try paragraphs, then sentences, then words)
# ============================================================
recursive = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],  # try these in order
    chunk_size=300,
    chunk_overlap=50,
)
show("4. Recursive (paragraph -> sentence -> word)", recursive.split_text(DOCUMENT))


print("""
============================================================
READING THIS:
  Fixed-size      — splits mid-word/mid-sentence. Fast, dumb, can split facts.
  Fixed+overlap   — overlap means a fact split at a boundary appears whole in
                    one of the two overlapping chunks. Costs duplication.
  Sentence-based  — respects sentence boundaries. More natural chunks, but
                    sizes vary and very long sentences still overflow.
  Recursive       — tries paragraph breaks first, falls back to sentence, then
                    word. Respects document STRUCTURE. The usual best default.

The tension to feel: smaller chunks = focused meaning but lost context;
larger chunks = self-contained but diluted meaning. Overlap buys back some
context at the cost of duplication. Recursive respects natural boundaries so
chunks break where the document already breaks.
============================================================
""")
