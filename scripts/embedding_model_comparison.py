"""Compare embedding models on the same retrieval task.

We measure three things per model:
  - dimension (storage cost proxy)
  - embedding speed (time to embed a batch)
  - retrieval quality (does it rank the RIGHT chunk first for known queries?)

Quality is measured with a tiny labeled set: for each query we KNOW which
chunk should rank #1. A good model puts it there.

Run: uv run python scripts/embedding_model_comparison.py
"""

from __future__ import annotations

import time

import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# A tiny labeled retrieval set — we KNOW the right answer
# ============================================================

# The "corpus" — chunks we'll retrieve from
CHUNKS = [
    "RAG retrieves relevant documents and adds them to the prompt before generation.",
    "Cosine similarity measures the angle between two vectors, ranging from -1 to 1.",
    "A circuit breaker stops sending requests to a failing service to let it recover.",
    "BPE tokenization merges frequent character pairs to build a subword vocabulary.",
    "pgvector is a Postgres extension that stores vectors and does similarity search.",
    "Rate limiting caps how many requests a client can make in a time window.",
]

# Queries paired with the index of the chunk that SHOULD rank #1
LABELED_QUERIES = [
    ("How does retrieval augmented generation work?", 0),
    ("What does cosine similarity compute?", 1),
    ("How do you protect against a downstream service that is down?", 2),
    ("How does subword tokenization build its vocabulary?", 3),
    ("How can I store embeddings in Postgres?", 4),
    ("How do I limit requests per client?", 5),
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def evaluate_model(model_name: str) -> dict:
    print(f"\nLoading {model_name}...")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    load_time = time.perf_counter() - t0

    # Embedding speed: embed all chunks, time it
    t0 = time.perf_counter()
    chunk_embeddings = model.encode(CHUNKS)
    embed_time = time.perf_counter() - t0

    dim = chunk_embeddings.shape[1]

    # Retrieval quality: for each query, does the right chunk rank #1?
    correct_at_1 = 0
    reciprocal_ranks = []
    for query, gold_idx in LABELED_QUERIES:
        q_emb = model.encode(query)
        sims = [cosine(q_emb, ce) for ce in chunk_embeddings]
        # Rank chunks by similarity, descending
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        rank_of_gold = ranked.index(gold_idx) + 1  # 1-based
        if rank_of_gold == 1:
            correct_at_1 += 1
        reciprocal_ranks.append(1.0 / rank_of_gold)

    precision_at_1 = correct_at_1 / len(LABELED_QUERIES)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return {
        "model": model_name,
        "dim": dim,
        "load_time_s": round(load_time, 2),
        "embed_time_ms": round(embed_time * 1000, 1),
        "precision_at_1": round(precision_at_1, 3),
        "mrr": round(mrr, 3),
    }


def main() -> None:
    models = [
        "all-MiniLM-L6-v2",  # 384-dim, fast baseline
        "BAAI/bge-small-en-v1.5",  # 384-dim, retrieval-tuned
        "BAAI/bge-base-en-v1.5",  # 768-dim, higher quality
    ]

    results = []
    for m in models:
        try:
            results.append(evaluate_model(m))
        except Exception as exc:
            print(f"  FAILED {m}: {exc}")

    print("\n" + "=" * 78)
    print(f"{'model':<28} {'dim':>5} {'load_s':>7} {'embed_ms':>9} {'P@1':>6} {'MRR':>6}")
    print("=" * 78)
    for r in results:
        print(
            f"{r['model']:<28} {r['dim']:>5} {r['load_time_s']:>7} "
            f"{r['embed_time_ms']:>9} {r['precision_at_1']:>6} {r['mrr']:>6}"
        )

    print("""
READING THIS:
  dim       — vector size. Storage + speed cost. Bigger isn't always better.
  load_s    — one-time model load (cold start at ingestion service boot).
  embed_ms  — time to embed 6 chunks. Scales linearly — multiply for your corpus.
  P@1       — fraction of queries where the RIGHT chunk ranked #1. Retrieval quality.
  MRR       — mean reciprocal rank. Rewards right chunk being near the top even if not #1.
""")


if __name__ == "__main__":
    main()
