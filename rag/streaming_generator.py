"""Streaming RAG: yield the answer token-by-token, emit citations at the end.

Event sequence (the SSE shape):
  sources    -> the chunks about to be used (immediate, before generation)
  token      -> each piece of the answer as it streams
  citations  -> resolved [n] markers (AFTER completion — can't resolve mid-stream)
  done       -> coverage + finished

Citations are inherently post-completion: you can't resolve a [n] marker until
the text containing it exists. That timing forces the event order.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import anthropic

from rag.cited_generator import (
    CITED_SYSTEM_PROMPT,
    _build_numbered_context,
    _coverage,
    _extract_markers,
    _retrieve_for_mode,
)

_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
GEN_MODEL = "claude-haiku-4-5-20251001"


def stream_answer(
    question: str,
    *,
    top_k: int = 5,
    mode: str = "rerank",
) -> Iterator[dict]:
    """Yield SSE-shaped events: sources, then tokens, then citations, then done."""
    chunks = _retrieve_for_mode(question, mode, top_k)

    # Event 1: sources — what we're about to ground in (immediate feedback)
    yield {
        "type": "sources",
        "chunks": [{"doc_id": c.doc_id, "snippet": c.text[:100]} for c in chunks],
    }

    if not chunks:
        yield {"type": "token", "text": "No relevant context was retrieved."}
        yield {"type": "citations", "citations": []}
        yield {"type": "done", "coverage": 0.0}
        return

    context = _build_numbered_context(chunks)
    user_message = f"""Context:
{context}

Question: {question}

Answer using only the numbered context above, citing chunk numbers in brackets."""

    # Event 2..N: tokens — stream the answer as it generates
    full_answer_parts: list[str] = []
    with _client.messages.stream(
        model=GEN_MODEL,
        max_tokens=600,
        system=CITED_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            full_answer_parts.append(text)
            yield {"type": "token", "text": text}

    answer = "".join(full_answer_parts)

    # NOW the answer is complete — resolve citations (post-completion)
    markers = _extract_markers(answer)
    citations: list[dict] = []
    for m in sorted(markers):
        if 1 <= m <= len(chunks):
            c = chunks[m - 1]
            citations.append(
                {
                    "marker": m,
                    "doc_id": c.doc_id,
                    "url": c.metadata.get("url", ""),
                }
            )

    # Event N+1: citations (resolved)
    yield {"type": "citations", "citations": citations}

    # Event N+2: done (with coverage)
    yield {"type": "done", "coverage": round(_coverage(answer), 3)}


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "How does prompt caching reduce costs?"
    mode = sys.argv[2] if len(sys.argv) > 2 else "rerank"

    print(f"Q: {q}  (mode: {mode})\n")
    print("--- STREAMING (SSE events) ---\n")

    for event in stream_answer(q, mode=mode):
        if event["type"] == "sources":
            print(f"[SOURCES] {len(event['chunks'])} chunks retrieved:")
            for ch in event["chunks"]:
                print(f"   - {ch['doc_id']}")
            print("\n[ANSWER] ", end="", flush=True)
        elif event["type"] == "token":
            print(event["text"], end="", flush=True)  # live typewriter
        elif event["type"] == "citations":
            print("\n\n[CITATIONS]")
            for c in event["citations"]:
                print(f"   [{c['marker']}] {c['doc_id']}")
        elif event["type"] == "done":
            print(f"\n[DONE] coverage={event['coverage']:.0%}")
