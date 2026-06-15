"""Embed sample chunks with bge-small and insert them into pgvector.

This is the ingestion phase of RAG in miniature: text -> embed -> store.
Run: uv run python rag/ingest_demo.py
"""

from __future__ import annotations

import json

import psycopg
from sentence_transformers import SentenceTransformer

DB_DSN = "postgresql://rag:rag@localhost:5436/rag"

# The same model chosen in Hour 2. MUST match the vector(384) column dimension.
MODEL_NAME = "BAAI/bge-small-en-v1.5"


# A tiny "corpus" — chunks about things you've built, so queries are meaningful
SAMPLE_CHUNKS = [
    {
        "doc_id": "rag-notes",
        "text": "RAG retrieves relevant document chunks and adds them to the prompt before the LLM generates an answer, grounding the response in real sources.",
    },
    {
        "doc_id": "rag-notes",
        "text": "An embedding converts text into a vector of numbers representing its meaning. Similar meanings produce vectors pointing in similar directions.",
    },
    {
        "doc_id": "vector-db",
        "text": "HNSW is a graph-based approximate nearest neighbor index. It uses express lanes for big jumps and local hops to home in, giving fast queries with high recall.",
    },
    {
        "doc_id": "vector-db",
        "text": "IVF clusters vectors into buckets and only searches the nearest buckets at query time. It uses less memory than HNSW but has lower recall.",
    },
    {
        "doc_id": "gateway",
        "text": "A circuit breaker stops sending requests to a failing provider after repeated failures, then probes for recovery after a cooldown period.",
    },
    {
        "doc_id": "gateway",
        "text": "Cost tracking logs every LLM call's tokens, USD cost, and latency to Postgres, making per-request economics queryable instead of guessed.",
    },
    {
        "doc_id": "embeddings",
        "text": "Cosine similarity measures the angle between two vectors, ranging from -1 to 1. It is the standard metric for comparing text embeddings.",
    },
    {
        "doc_id": "embeddings",
        "text": "pgvector is a Postgres extension adding a vector column type and similarity search operators, so vectors live alongside relational data.",
    },
]


def main() -> None:
    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # Embed all chunk texts in one batch (efficient)
    texts = [c["text"] for c in SAMPLE_CHUNKS]
    embeddings = model.encode(texts, normalize_embeddings=True)
    print(f"Embedded {len(texts)} chunks, dim={embeddings.shape[1]}")

    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            # Clear any prior demo rows so re-runs are clean
            cur.execute("TRUNCATE chunks RESTART IDENTITY;")

            for chunk, emb in zip(SAMPLE_CHUNKS, embeddings, strict=False):
                # pgvector accepts a vector as a string like '[0.1,0.2,...]'
                emb_str = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
                cur.execute(
                    """
                    INSERT INTO chunks (doc_id, chunk_text, metadata, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        chunk["doc_id"],
                        chunk["text"],
                        json.dumps({"source": chunk["doc_id"]}),
                        emb_str,
                    ),
                )
        conn.commit()
    print(f"Inserted {len(SAMPLE_CHUNKS)} chunks into pgvector.")


if __name__ == "__main__":
    main()
