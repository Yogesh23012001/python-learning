"""Hybrid search: fuse dense (vector) and sparse (BM25) retrieval via
Reciprocal Rank Fusion.

RRF combines rankings by POSITION, not raw score — which sidesteps the
problem that cosine similarity (0-1) and BM25 scores (unbounded) live on
different scales. A chunk that ranks well in either retriever surfaces;
one that ranks well in both surfaces highest.
"""

from __future__ import annotations

from rag.bm25_search import bm25_search
from rag.pipeline import RetrievedChunk
from rag.retriever import retrieve as dense_retrieve

RRF_K = 60  # standard damping constant


def _chunk_key(rc: RetrievedChunk) -> str:
    """Identity for dedup across the two result lists."""
    return f"{rc.doc_id}::{hash(rc.text)}"


def hybrid_search(
    query: str,
    *,
    top_k: int = 5,
    retrieve_n: int = 10,  # pull more from each retriever before fusing
) -> list[RetrievedChunk]:
    """Retrieve from both, fuse with RRF, return top_k."""
    dense_results = dense_retrieve(query, top_k=retrieve_n)
    sparse_results = bm25_search(query, top_k=retrieve_n)

    # Build RRF scores
    rrf_scores: dict[str, float] = {}
    chunk_by_key: dict[str, RetrievedChunk] = {}

    for rank, rc in enumerate(dense_results, 1):
        key = _chunk_key(rc)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        chunk_by_key[key] = rc

    for rank, rc in enumerate(sparse_results, 1):
        key = _chunk_key(rc)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        chunk_by_key.setdefault(key, rc)

    # Rank by fused score
    ranked_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]

    return [
        RetrievedChunk(
            doc_id=chunk_by_key[key].doc_id,
            text=chunk_by_key[key].text,
            score=rrf_scores[key],  # fused RRF score
            metadata=chunk_by_key[key].metadata,
        )
        for key in ranked_keys
    ]


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "cache_control ephemeral"

    print(f"Query: {q!r}\n")
    print("--- DENSE only ---")
    for i, rc in enumerate(dense_retrieve(q, top_k=5), 1):
        print(f"  {i}. [{rc.doc_id}] {rc.text[:80]}...")
    print("\n--- BM25 only ---")
    for i, rc in enumerate(bm25_search(q, top_k=5), 1):
        print(f"  {i}. [{rc.doc_id}] {rc.text[:80]}...")
    print("\n--- HYBRID (RRF) ---")
    for i, rc in enumerate(hybrid_search(q, top_k=5), 1):
        print(f"  {i}. [{rc.doc_id}] rrf={rc.score:.4f} {rc.text[:80]}...")
