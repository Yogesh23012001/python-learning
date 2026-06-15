"""Query pgvector: embed a question, retrieve the top-k nearest chunks.

This is the retrieval phase of RAG. The <=> operator is cosine distance;
ORDER BY it ASC gives nearest-first. The HNSW index makes this fast.

Run: uv run python rag/query_demo.py "your question here"
"""

from __future__ import annotations

import sys

import psycopg
from sentence_transformers import SentenceTransformer

DB_DSN = "postgresql://rag:rag@localhost:5436/rag"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "How does HNSW make search fast?"
    top_k = 3

    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode(question, normalize_embeddings=True)
    q_str = "[" + ",".join(f"{x:.6f}" for x in q_emb) + "]"

    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                doc_id,
                chunk_text,
                embedding <=> %s AS cosine_distance
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (q_str, q_str, top_k),
        )
        rows = cur.fetchall()

    print(f"\nQuestion: {question}\n")
    print(f"Top {top_k} chunks (lower distance = more similar):\n")
    for i, (doc_id, text, dist) in enumerate(rows, 1):
        similarity = 1 - dist  # cosine distance -> cosine similarity
        print(f"{i}. [{doc_id}] similarity={similarity:.3f}")
        print(f"   {text}\n")


if __name__ == "__main__":
    main()
