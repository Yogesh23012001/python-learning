"""Citation-aware generation: the model cites [n] markers referencing the
numbered context chunks; we parse and resolve them to source chunks.

Builds on the design from Hour 1: model-generated inline citations, with each
[n] resolved to a real retrieved chunk (doc_id, url, snippet). Includes the
defensive check for markers that don't map to a retrieved chunk — itself a
groundedness signal (the model citing a source it wasn't given).
"""

from __future__ import annotations

import os
import re

import anthropic

from rag.contextual_search import contextual_retrieve
from rag.hybrid_search import hybrid_search
from rag.hyde import hyde_retrieve
from rag.pipeline import Citation, CitedAnswer, RetrievedChunk
from rag.reranker import rerank
from rag.retriever import retrieve

_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
GEN_MODEL = "claude-haiku-4-5-20251001"


CITED_SYSTEM_PROMPT = """You are a question-answering assistant for the Anthropic documentation. \
Answer using ONLY the provided numbered context chunks. Rules:
- Base every claim solely on the context. Do not use prior knowledge.
- After each claim, cite the chunk number(s) it came from in square brackets, e.g. [1] or [2][3].
- Every sentence that states a fact from the context MUST carry a citation.
- If the context does not contain enough information, say so clearly and do not guess.
- Cite ONLY chunk numbers that appear in the provided context. Never invent a citation number."""


def _build_numbered_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (source: {c.doc_id})\n{c.text}")
    return "\n\n".join(parts)


def _retrieve_for_mode(question: str, mode: str, top_k: int) -> list[RetrievedChunk]:
    if mode == "rerank":
        return rerank(question, hybrid_search(question, top_k=20), top_k=top_k)
    if mode == "hybrid":
        return hybrid_search(question, top_k=top_k)
    if mode == "hyde":
        return hyde_retrieve(question, top_k=top_k)
    if mode == "contextual":
        return contextual_retrieve(question, top_k=top_k)
    return retrieve(question, top_k=top_k)


# Match [1], [2][3], [1, 2] style markers
_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _extract_markers(answer: str) -> set[int]:
    """Pull all cited chunk numbers out of the answer text."""
    markers: set[int] = set()
    for match in _MARKER_RE.finditer(answer):
        for num in match.group(1).split(","):
            markers.add(int(num.strip()))
    return markers


def _coverage(answer: str) -> float:
    """Fraction of sentences that carry at least one [n] citation."""
    # Split into sentences (rough — period/question/exclamation followed by space)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if _MARKER_RE.search(s))
    return cited / len(sentences)


def answer_with_citations(
    question: str,
    *,
    top_k: int = 5,
    mode: str = "dense",
) -> CitedAnswer:
    """Full RAG flow with inline citations resolved to source chunks."""
    chunks = _retrieve_for_mode(question, mode, top_k)
    if not chunks:
        return CitedAnswer(question, "No relevant context was retrieved.", [], [], 0.0)

    context = _build_numbered_context(chunks)
    user_message = f"""Context:
{context}

Question: {question}

Answer using only the numbered context above, citing chunk numbers in brackets."""

    response = _client.messages.create(
        model=GEN_MODEL,
        max_tokens=600,
        system=CITED_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    answer = "".join(b.text for b in response.content if b.type == "text")

    # Resolve markers -> citations. Defensive: a marker must map to a real chunk.
    markers = _extract_markers(answer)
    citations: list[Citation] = []
    hallucinated_markers: list[int] = []
    for m in sorted(markers):
        if 1 <= m <= len(chunks):
            c = chunks[m - 1]
            citations.append(
                Citation(
                    marker=m,
                    doc_id=c.doc_id,
                    url=c.metadata.get("url", ""),
                    snippet=c.text[:150],
                )
            )
        else:
            # The model cited a chunk number it wasn't given — a groundedness red flag
            hallucinated_markers.append(m)

    if hallucinated_markers:
        print(f"  ⚠️  Model cited non-existent chunks: {hallucinated_markers}")

    return CitedAnswer(
        question=question,
        answer=answer,
        citations=citations,
        retrieved=chunks,
        coverage=round(_coverage(answer), 3),
    )


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "How does prompt caching reduce costs?"
    mode = sys.argv[2] if len(sys.argv) > 2 else "rerank"

    result = answer_with_citations(q, mode=mode)
    print(f"\nQ: {result.question}  (mode: {mode})\n")
    print(f"A: {result.answer}\n")
    print(f"Citation coverage: {result.coverage:.0%} of sentences cited\n")
    print("Citations:")
    for c in result.citations:
        print(f"  [{c.marker}] {c.doc_id}")
        print(f"      {c.snippet}...")
