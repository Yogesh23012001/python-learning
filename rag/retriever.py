"""Retrieval: embed a question, return the top-k most similar chunks from pgvector."""

from __future__ import annotations

import psycopg
from sentence_transformers import SentenceTransformer

from rag.pipeline import RetrievedChunk

DB_DSN = "postgresql://rag:rag@localhost:5436/rag"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Load the model once at import — reused across queries (same pattern as the
# semantic cache: load once, embed many).
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def retrieve(question: str, *, top_k: int = 5) -> list[RetrievedChunk]:
    """Embed the question, return the top_k nearest chunks by cosine similarity."""
    model = _get_model()
    q_emb = model.encode(question, normalize_embeddings=True)
    q_str = "[" + ",".join(f"{x:.6f}" for x in q_emb) + "]"

    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT doc_id, chunk_text, metadata,
                   1 - (embedding <=> %s) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (q_str, q_str, top_k),
        )
        rows = cur.fetchall()

    return [
        RetrievedChunk(
            doc_id=doc_id,
            text=chunk_text,
            score=float(similarity),
            metadata=metadata,
        )
        for (doc_id, chunk_text, metadata, similarity) in rows
    ]


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "How do I use XML tags?"
    for i, rc in enumerate(retrieve(q, top_k=5), 1):
        print(f"{i}. [{rc.doc_id}] score={rc.score:.3f}")
        print(f"   {rc.text[:120]}...\n")
