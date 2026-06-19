"""HyDE — Hypothetical Document Embeddings.

Instead of embedding the question, generate a hypothetical ANSWER and embed
that. A fake answer is answer-shaped, so it lands near real answer chunks in
vector space — bridging the question-space vs answer-space mismatch.

The hypothetical answer is a retrieval PROBE: its vocabulary finds the right
chunks; its facts are never used. The real answer comes from retrieved chunks.
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from rag.pipeline import RetrievedChunk

DB_DSN = "postgresql://rag:rag@localhost:5436/rag"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Load the project .env (one dir up from rag/) so ANTHROPIC_API_KEY is available
# when this module runs standalone — the same key source the evals use.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit("ANTHROPIC_API_KEY not set in .env")

# The SDK reads ANTHROPIC_API_KEY from the environment automatically.
_client = anthropic.Anthropic()
HYDE_MODEL = "claude-haiku-4-5-20251001"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


# The prompt that generates the hypothetical answer. Key: ask for a SHORT,
# documentation-style answer — we want answer-SHAPED text in the docs' register,
# not a chatty response. Accuracy doesn't matter; shape and vocabulary do.
HYDE_PROMPT = """Write a short, factual paragraph (2-4 sentences) that directly answers the question \
as if it were an excerpt from technical API documentation. Use the precise, declarative style of \
reference documentation. Do not hedge, do not say "I think" — just state the answer plainly, even \
if you are unsure. The goal is a documentation-style passage, not a conversation.

Question: {question}

Documentation excerpt:"""


def generate_hypothetical(question: str) -> str:
    """Ask the LLM for a documentation-style hypothetical answer."""
    response = _client.messages.create(
        model=HYDE_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(question=question)}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def hyde_retrieve(
    question: str,
    *,
    top_k: int = 5,
    return_hypothetical: bool = False,
) -> list[RetrievedChunk] | tuple[list[RetrievedChunk], str]:
    """Generate a hypothetical answer, embed IT, retrieve chunks similar to it."""
    hypothetical = generate_hypothetical(question)

    model = _get_model()
    # Embed the HYPOTHETICAL ANSWER, not the question — the whole point of HyDE
    h_emb = model.encode(hypothetical, normalize_embeddings=True)
    h_str = "[" + ",".join(f"{x:.6f}" for x in h_emb) + "]"

    with psycopg.connect(DB_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT doc_id, chunk_text, metadata,
                   1 - (embedding <=> %s) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (h_str, h_str, top_k),
        )
        rows = cur.fetchall()

    chunks = [RetrievedChunk(doc_id=d, text=t, score=float(s), metadata=m) for (d, t, m, s) in rows]
    if return_hypothetical:
        return chunks, hypothetical
    return chunks


if __name__ == "__main__":
    import sys

    q = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Why does Claude keep forgetting what we talked about earlier in a long chat?"
    )

    chunks, hypothetical = hyde_retrieve(q, top_k=5, return_hypothetical=True)
    print(f"Question: {q!r}\n")
    print(f"--- Hypothetical answer (the retrieval probe) ---\n{hypothetical}\n")
    print("--- Chunks retrieved via the hypothetical ---")
    for i, rc in enumerate(chunks, 1):
        print(f"  {i}. [{rc.doc_id}] score={rc.score:.3f}")
        print(f"     {rc.text[:90]}...")
