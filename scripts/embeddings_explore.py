"""Explore embeddings: generate vectors, measure semantic similarity.

The foundation for semantic caching and RAG. Run after `uv add sentence-transformers`.

Run: uv run python scripts/embeddings_explore.py
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# all-MiniLM-L6-v2: 384-dim, fast, good enough for learning + semantic cache.
# First run downloads ~80MB.
print("Loading model (first run downloads ~80MB)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity: dot product of unit vectors. Range -1 to 1."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ============================================================
# 1. What an embedding actually IS
# ============================================================
sentence = "What is karpathy's GitHub score?"
embedding = model.encode(sentence)
print("=" * 60)
print("WHAT AN EMBEDDING IS")
print("=" * 60)
print(f"Text: {sentence!r}")
print(f"Embedding shape: {embedding.shape}")  # (384,)
print(f"First 8 numbers: {embedding[:8]}")
print(f"(It's just {embedding.shape[0]} floats representing the meaning.)\n")


# ============================================================
# 2. Semantic similarity — the whole point
# ============================================================
print("=" * 60)
print("SEMANTIC SIMILARITY")
print("=" * 60)

# These three ask the SAME thing in different words
q1 = "What is karpathy's GitHub score?"
q2 = "Tell me karpathy's developer score on GitHub"
q3 = "How highly rated is karpathy as a GitHub developer?"

# This one is unrelated
q4 = "What's the weather in Tokyo today?"

# This one shares words but different meaning
q5 = "What is the score of the cricket match?"

emb = {q: model.encode(q) for q in [q1, q2, q3, q4, q5]}

print(f"\nBase query: {q1!r}\n")
for q in [q2, q3, q4, q5]:
    sim = cosine_similarity(emb[q1], emb[q])
    marker = (
        "← SAME meaning"
        if sim > 0.6
        else ("← shares words, diff meaning" if sim > 0.4 else "← unrelated")
    )
    print(f"  sim={sim:.3f}  {q!r}")
    print(f"           {marker}\n")


# ============================================================
# 3. Word-level intuition
# ============================================================
print("=" * 60)
print("WORD-LEVEL INTUITION")
print("=" * 60)

pairs = [
    ("dog", "puppy"),  # near-synonyms
    ("dog", "cat"),  # same category
    ("dog", "database"),  # unrelated
    ("Python", "JavaScript"),  # both languages
    ("Python", "banana"),  # unrelated
]
print()
for a, b in pairs:
    sim = cosine_similarity(model.encode(a), model.encode(b))
    print(f"  sim={sim:.3f}  {a!r} <-> {b!r}")


# ============================================================
# 4. The semantic cache threshold question
# ============================================================
print("\n" + "=" * 60)
print("SEMANTIC CACHE: what threshold catches paraphrases but not different questions?")
print("=" * 60)
print("""
For a semantic cache, you pick a similarity THRESHOLD:
  - Above threshold -> treat as the same query, return cached answer
  - Below threshold -> treat as new query, call the LLM

From the numbers above:
  - Paraphrases of the same question scored ~0.6-0.85
  - Shares-words-but-different scored ~0.4-0.5
  - Unrelated scored < 0.3

A threshold around 0.80-0.85 is conservative (only near-identical meaning hits
cache). Lower (0.70) catches more paraphrases but risks false cache hits.
This threshold is the single most important tuning knob in a semantic cache.
""")
