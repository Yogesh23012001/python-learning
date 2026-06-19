"""Re-ingest: add document-aware context to each chunk, embed the contextualized
version, store in the contextual_embedding column.

Processes chunks grouped BY DOCUMENT so prompt caching works — all chunks of a
document are done consecutively while that document is cached.
"""

from __future__ import annotations

import time

import psycopg
from sentence_transformers import SentenceTransformer

from rag.chunker import chunk_documents
from rag.contextualizer import contextualize_chunk
from rag.loader import load_corpus

DB_DSN = "postgresql://rag:rag@localhost:5436/rag"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main() -> None:
    print("Loading corpus + chunking...")
    docs = load_corpus()
    chunks = chunk_documents(docs)
    print(f"{len(chunks)} chunks to contextualize.")

    # Map doc_id -> full document text (for the context prompt)
    doc_text_by_id = {d.doc_id: d.text for d in docs}

    model = SentenceTransformer(MODEL_NAME)

    # Process grouped by doc_id so the cached document stays hot
    chunks_by_doc: dict[str, list] = {}
    for c in chunks:
        chunks_by_doc.setdefault(c.doc_id, []).append(c)

    t0 = time.perf_counter()
    done = 0

    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        # We need to match contextualized chunks back to stored rows.
        # Simplest: rebuild the table content in order. Since ingest.py
        # inserted chunks in the same order, we re-fetch ids in order.
        cur.execute("SELECT id, doc_id, chunk_text FROM chunks ORDER BY id")
        rows = cur.fetchall()

        # Build a lookup: (doc_id, chunk_text) -> id
        id_lookup = {(doc_id, text): cid for cid, doc_id, text in rows}

        for doc_id, doc_chunks in chunks_by_doc.items():
            whole_doc = doc_text_by_id[doc_id]
            for c in doc_chunks:
                # Generate context (document cached across this doc's chunks)
                context = contextualize_chunk(whole_doc, c.text)
                contextualized = f"{context}\n\n{c.text}"

                # Embed the CONTEXTUALIZED text
                emb = model.encode(contextualized, normalize_embeddings=True)
                emb_str = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"

                cid = id_lookup.get((c.doc_id, c.text))
                if cid is None:
                    continue  # safety: skip if no match
                cur.execute(
                    "UPDATE chunks SET context_text = %s, contextual_embedding = %s WHERE id = %s",
                    (context, emb_str, cid),
                )
                done += 1
                if done % 50 == 0:
                    elapsed = time.perf_counter() - t0
                    print(f"  {done}/{len(chunks)} contextualized ({elapsed:.0f}s)")
            conn.commit()  # commit per document

    elapsed = time.perf_counter() - t0
    print(f"\nDone. Contextualized {done} chunks in {elapsed:.0f}s.")


if __name__ == "__main__":
    main()
