"""The RAG pipeline — composable stages.

Ingestion:  load -> chunk -> embed -> store
Query:      embed -> retrieve -> generate

Each stage is a clear function. This file is the skeleton; stages get filled
in across Hours 2-3. Defining the shape first keeps the pipeline clean — the
same 'define the seams before the logic' discipline from the gateway.
"""

from __future__ import annotations

from dataclasses import dataclass

# ============================================================
# Shared types — the pipeline's vocabulary
# ============================================================


@dataclass(frozen=True)
class Chunk:
    """A piece of a document, ready to embed."""

    doc_id: str
    chunk_index: int  # position within the document
    text: str
    metadata: dict


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by retrieval, with its similarity score."""

    doc_id: str
    text: str
    score: float  # similarity (higher = more relevant)
    metadata: dict


@dataclass(frozen=True)
class RAGAnswer:
    """The final output: answer + the chunks it was grounded in."""

    question: str
    answer: str
    retrieved: list[RetrievedChunk]  # what was retrieved (for citations later)


# ============================================================
# Pipeline stages (signatures now, bodies across H2-3)
# ============================================================

# Ingestion:
#   load_corpus()           -> list[LoadedDocument]   (loader.py, done)
#   chunk_documents(docs)   -> list[Chunk]            (H2)
#   embed_and_store(chunks) -> None                   (H2)
#
# Query:
#   retrieve(question, k)   -> list[RetrievedChunk]   (H2)
#   generate(question, chunks) -> RAGAnswer           (H3)
#   answer_question(question)  -> RAGAnswer           (H3, ties it together)
