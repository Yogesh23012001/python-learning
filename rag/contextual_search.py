"""Contextual retrieval: query against the contextual_embedding column."""

from __future__ import annotations

import psycopg
from sentence_transformers import SentenceTransformer

from rag.pipeline import RetrievedChunk

DB_DSN = "postgresql://rag:rag@localhost:5436/rag"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def contextual_retrieve(question: str, *, top_k: int = 5) -> list[RetrievedChunk]:
    """Retrieve against the contextualized embeddings."""
    model = _get_model()
    q_emb = model.encode(question, normalize_embeddings=True)
    q_str = "[" + ",".join(f"{x:.6f}" for x in q_emb) + "]"

    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT doc_id, chunk_text, metadata,
                   1 - (contextual_embedding <=> %s) AS similarity
            FROM chunks
            WHERE contextual_embedding IS NOT NULL
            ORDER BY contextual_embedding <=> %s
            LIMIT %s
            """,
            (q_str, q_str, top_k),
        )
        rows = cur.fetchall()

    return [RetrievedChunk(doc_id=d, text=t, score=float(s), metadata=m) for (d, t, m, s) in rows]
