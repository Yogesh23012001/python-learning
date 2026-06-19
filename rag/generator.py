"""Generation: build a grounded prompt from retrieved chunks, call Anthropic.

The prompt is the craft. Its job: answer using ONLY the retrieved context,
and explicitly say when the context doesn't contain the answer. This is what
makes RAG faithful — the model is instructed to ground in retrieval, not
its training memory.
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from rag.contextual_search import contextual_retrieve
from rag.hybrid_search import hybrid_search
from rag.hyde import hyde_retrieve
from rag.pipeline import RAGAnswer, RetrievedChunk
from rag.reranker import rerank
from rag.retriever import retrieve

# Load the project .env (one dir up from rag/) so ANTHROPIC_API_KEY is available
# when this module runs standalone — the same key source the evals use.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit("ANTHROPIC_API_KEY not set in .env")

# The SDK reads ANTHROPIC_API_KEY from the environment automatically.
_client = anthropic.Anthropic()
GEN_MODEL = "claude-haiku-4-5-20251001"


SYSTEM_PROMPT = """You are a question-answering assistant for the Anthropic documentation. \
Answer the user's question using ONLY the provided context chunks. \
Follow these rules strictly:
- Base your answer solely on the context. Do not use prior knowledge.
- If the context does not contain enough information to answer, say so clearly: \
"The provided context doesn't contain enough information to answer this." Do not guess.
- Be concise and direct. Quote or closely paraphrase the relevant context.
- Do not mention "the context" or "the chunks" in a distracting way — just answer naturally \
as if you know it, but grounded in what was provided."""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Chunk {i}] (source: {c.doc_id})\n{c.text}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[RetrievedChunk]) -> str:
    """Call the LLM with the question grounded in retrieved chunks."""
    if not chunks:
        return "No relevant context was retrieved for this question."

    context = _build_context(chunks)
    user_message = f"""Context:
{context}

Question: {question}

Answer using only the context above."""

    response = _client.messages.create(
        model=GEN_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def answer_question(
    question: str,
    *,
    top_k: int = 5,
    mode: str = "dense",  # "dense" | "hybrid" | "rerank" | "hyde"
) -> RAGAnswer:
    """Full RAG flow.
    mode:
      'dense'  - vector retrieval only
      'hybrid' - dense + BM25 (RRF)
      'rerank' - hybrid retrieve broad (top 20) -> cross-encoder rerank -> top_k
      'hyde'   - embed a hypothetical answer, retrieve chunks similar to it
    """
    if mode == "rerank":
        candidates = hybrid_search(question, top_k=20)
        chunks = rerank(question, candidates, top_k=top_k)
    elif mode == "hybrid":
        chunks = hybrid_search(question, top_k=top_k)
    elif mode == "hyde":
        chunks = hyde_retrieve(question, top_k=top_k)
    elif mode == "contextual":
        chunks = contextual_retrieve(question, top_k=top_k)
    else:
        chunks = retrieve(question, top_k=top_k)

    answer = generate(question, chunks)
    return RAGAnswer(question=question, answer=answer, retrieved=chunks)


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "How do I use XML tags in prompts?"
    mode = sys.argv[2] if len(sys.argv) > 2 else "dense"
    result = answer_question(q, mode=mode)

    print(f"\nQ: {result.question}  (mode: {mode})\n")
    print(f"A: {result.answer}\n")
    print("=" * 60)
    print(f"Grounded in {len(result.retrieved)} chunks:")
    for i, c in enumerate(result.retrieved, 1):
        print(f"  {i}. [{c.doc_id}] score={c.score:.3f}")
