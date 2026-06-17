"""Full ingestion: load -> chunk -> embed -> store in pgvector.

This is the complete ingestion phase. Run once per corpus. After this, the
~172 chunks are searchable via the retriever.
"""

from __future__ import annotations

import json

import psycopg
from sentence_transformers import SentenceTransformer

from rag.chunker import chunk_documents
from rag.loader import load_corpus

DB_DSN = "postgresql://rag:rag@localhost:5436/rag"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_BATCH = 64  # embed in batches for efficiency


def main() -> None:
    print("Loading corpus...")
    docs = load_corpus()
    print(f"\nChunking {len(docs)} docs...")
    chunks = chunk_documents(docs)
    print(f"{len(chunks)} chunks to embed.")

    print(f"\nLoading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding chunks (batched)...")
    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
    )
    print(f"Embedded {len(embeddings)} chunks, dim={embeddings.shape[1]}")

    print("\nStoring in pgvector...")
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE chunks RESTART IDENTITY;")
            for chunk, emb in zip(chunks, embeddings, strict=False):
                emb_str = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
                cur.execute(
                    """
                    INSERT INTO chunks (doc_id, chunk_text, metadata, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (chunk.doc_id, chunk.text, json.dumps(chunk.metadata), emb_str),
                )
        conn.commit()

    print(f"Stored {len(chunks)} chunks. Ingestion complete.")


if __name__ == "__main__":
    main()
