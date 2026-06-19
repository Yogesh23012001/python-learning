"""Contextual retrieval: prepend LLM-generated, document-aware context to each
chunk before embedding.

Uses Anthropic's actual contextualization prompt. The whole document is sent
with EVERY chunk of that document — so we cache the document (cache_control on
the document block). First chunk of a doc writes the cache; every subsequent
chunk reads it at ~10% cost. This is the trick that makes per-chunk LLM calls
affordable.
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Load the project .env (one dir up from rag/) so ANTHROPIC_API_KEY is available
# when this module runs standalone — the same key source the evals use.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit("ANTHROPIC_API_KEY not set in .env")

# The SDK reads ANTHROPIC_API_KEY from the environment automatically.
_client = anthropic.Anthropic()
CONTEXT_MODEL = "claude-haiku-4-5-20251001"

# Anthropic's published contextualization prompt.
DOCUMENT_CONTEXT_PROMPT = """<document>
{whole_document}
</document>"""

CHUNK_CONTEXT_PROMPT = """Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document \
for the purposes of improving search retrieval of the chunk. Answer only with the succinct \
context and nothing else."""


def contextualize_chunk(whole_document: str, chunk_content: str) -> str:
    """Generate a short context snippet situating the chunk in its document.

    The document block is marked cache_control=ephemeral so that repeated calls
    for chunks of the SAME document reuse the cached document prefix.
    """
    response = _client.messages.create(
        model=CONTEXT_MODEL,
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": DOCUMENT_CONTEXT_PROMPT.format(whole_document=whole_document),
                        "cache_control": {"type": "ephemeral"},  # <-- the cost trick
                    },
                    {
                        "type": "text",
                        "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk_content),
                    },
                ],
            }
        ],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


if __name__ == "__main__":
    # Quick demo on one chunk
    doc = "Prompt caching lets you cache portions of prompts. Cache reads cost 10% of base input price. Use cache_control to mark cacheable blocks."
    chunk = 'cache_control={"type": "ephemeral"}'
    ctx = contextualize_chunk(doc, chunk)
    print(f"Chunk: {chunk}")
    print(f"Generated context: {ctx}")
