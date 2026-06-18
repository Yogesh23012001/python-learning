"""Reranking: a cross-encoder second stage that re-scores retrieved candidates.

First-stage retrieval (bi-encoder) embeds query and chunk SEPARATELY — fast,
scales to the whole corpus, but only captures topical similarity. The reranker
(cross-encoder) reads query and chunk TOGETHER, judging true relevance. Too
slow for the whole corpus, perfect for re-scoring a handful of candidates.

Pattern: retrieve broad (top 20) -> rerank precise (top 5) -> generate.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag.pipeline import RetrievedChunk

# A small, fast, local cross-encoder reranker. No API key.
RERANKER_MODEL = "BAAI/bge-reranker-base"

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    *,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Re-score candidates with the cross-encoder, return the top_k by relevance."""
    if not candidates:
        return []

    model = _get_reranker()
    # The cross-encoder scores (query, chunk_text) PAIRS — query and chunk
    # read together, the whole point of a cross-encoder.
    pairs = [(query, c.text) for c in candidates]
    scores = model.predict(pairs)  # higher = more relevant

    # Attach new scores, sort descending, take top_k
    rescored = [
        RetrievedChunk(
            doc_id=c.doc_id,
            text=c.text,
            score=float(s),  # the reranker's relevance score
            metadata=c.metadata,
        )
        for c, s in zip(candidates, scores, strict=False)
    ]
    rescored.sort(key=lambda rc: rc.score, reverse=True)
    return rescored[:top_k]


if __name__ == "__main__":
    import sys

    from rag.hybrid_search import hybrid_search

    q = sys.argv[1] if len(sys.argv) > 1 else "How does Claude decide whether to call a tool?"

    # Stage 1: retrieve broad
    candidates = hybrid_search(q, top_k=20)
    print(f"Query: {q!r}\n")
    print("--- BEFORE rerank (hybrid top-20, showing top 6) ---")
    for i, rc in enumerate(candidates[:6], 1):
        print(f"  {i}. [{rc.doc_id}] {rc.text[:75]}...")

    # Stage 2: rerank precise
    reranked = rerank(q, candidates, top_k=6)
    print("\n--- AFTER rerank (cross-encoder top 6) ---")
    for i, rc in enumerate(reranked, 1):
        print(f"  {i}. [{rc.doc_id}] rerank={rc.score:.3f} {rc.text[:75]}...")
