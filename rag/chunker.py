"""Chunk loaded documents into embeddable pieces.

Uses recursive splitting (respects paragraph -> sentence -> word boundaries,
the Monday pick) with overlap (so a fact split at a boundary survives whole
in one of the two overlapping chunks).
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.loader import LoadedDocument
from rag.pipeline import Chunk

# Monday's starting values. These are HYPERPARAMETERS — tuned in Thursday's eval.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


def chunk_documents(
    docs: list[LoadedDocument],
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split each document into overlapping chunks, preserving doc_id + position."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    all_chunks: list[Chunk] = []
    for doc in docs:
        pieces = splitter.split_text(doc.text)
        for i, piece in enumerate(pieces):
            # Skip chunks that are too tiny to carry meaning (the starved-chunk
            # guard from Monday — respect-boundaries and good-size are separate knobs)
            if len(piece.strip()) < 50:
                continue
            all_chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    chunk_index=i,
                    text=piece.strip(),
                    metadata={
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "url": doc.url,
                        "chunk_index": i,
                    },
                )
            )
    return all_chunks


if __name__ == "__main__":
    from rag.loader import load_corpus

    docs = load_corpus()
    chunks = chunk_documents(docs)

    sizes = [len(c.text) for c in chunks]
    print(f"\n{len(chunks)} chunks from {len(docs)} docs")
    print(f"chunk size: avg={sum(sizes)//len(sizes)} min={min(sizes)} max={max(sizes)}")
    # Show how one doc chunked
    first_doc_chunks = [c for c in chunks if c.doc_id == chunks[0].doc_id]
    print(f"\nFirst doc ({chunks[0].doc_id}) → {len(first_doc_chunks)} chunks. First two:")
    for c in first_doc_chunks[:2]:
        print(f"\n  [{c.chunk_index}] ({len(c.text)} chars)")
        print(f"  {c.text[:150]!r}")
