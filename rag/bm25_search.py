"""BM25 (sparse/keyword) search over chunk text.

Complements dense vector search: BM25 matches exact tokens (API params,
acronyms, identifiers) that dense embedding search fuzzes. Built in-memory
with rank_bm25 — for production scale, Postgres full-text search is the
equivalent inside the database.
"""

from __future__ import annotations

import re

import psycopg
from rank_bm25 import BM25Okapi

from rag.pipeline import RetrievedChunk

DB_DSN = "postgresql://rag:rag@localhost:5436/rag"


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumerics, keep underscores.

    Keeping underscores matters — 'tool_choice' and 'cache_control' are single
    meaningful tokens, not 'tool' + 'choice'.
    """
    return re.findall(r"[a-z0-9_]+", text.lower())


class BM25Index:
    """In-memory BM25 index over all chunks. Built once from pgvector."""

    def __init__(self) -> None:
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._build()

    def _build(self) -> None:
        with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
            cur.execute("SELECT doc_id, chunk_text, metadata FROM chunks ORDER BY id")
            for doc_id, text, metadata in cur.fetchall():
                self._chunks.append({"doc_id": doc_id, "text": text, "metadata": metadata})

        tokenized_corpus = [_tokenize(c["text"]) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        assert self._bm25 is not None
        scores = self._bm25.get_scores(_tokenize(query))
        # Rank chunk indices by BM25 score, descending
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                doc_id=self._chunks[i]["doc_id"],
                text=self._chunks[i]["text"],
                score=float(scores[i]),
                metadata=self._chunks[i]["metadata"],
            )
            for i in ranked
            if scores[i] > 0  # drop zero-score (no query terms matched)
        ]


# Module-level singleton — build the index once
_index: BM25Index | None = None


def bm25_search(query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
    global _index
    if _index is None:
        _index = BM25Index()
    return _index.search(query, top_k=top_k)


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "tool_choice auto"
    print(f"BM25 results for: {q!r}\n")
    for i, rc in enumerate(bm25_search(q, top_k=5), 1):
        print(f"{i}. [{rc.doc_id}] bm25={rc.score:.2f}")
        print(f"   {rc.text[:120]}...\n")
