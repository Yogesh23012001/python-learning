"""Semantic cache for LLM responses — hits on meaning, not exact bytes.

Uses sentence-transformers embeddings + cosine similarity. A query hits the
cache if its embedding is within THRESHOLD of any stored prompt's embedding.

Storage: Redis. Each entry stores the embedding + prompt + serialized response.
Lookup is brute-force (scan all entries, compute similarity). This is O(N) per
query — fine for a small cache. At scale (thousands of entries), replace the
brute-force scan with a Redis vector index (RediSearch) or pgvector.

Limitation: embeddings are computed with all-MiniLM-L6-v2 (local, 384-dim).
A query and a cached prompt about DIFFERENT users could be semantically close
("what is X's score" vs "what is Y's score" — same shape, different subject) and
falsely hit. The threshold must be tuned to avoid this. See SIMILARITY_THRESHOLD.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
import redis.asyncio as aioredis
import structlog
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from llm.router import RoutedLLMResponse

logger = structlog.get_logger(__name__)

# Tuned from the embeddings_explore.py experiment:
#   paraphrases scored 0.81-0.91, "shares words diff meaning" scored 0.34.
# 0.85 is conservative — only near-identical meaning hits. Raise to reduce
# false hits, lower to catch more paraphrases.
SIMILARITY_THRESHOLD = 0.85

_SEMANTIC_INDEX_KEY = "llmcache:semantic:index"  # Redis SET of entry ids
_SEMANTIC_ENTRY_PREFIX = "llmcache:semantic:entry:"


class SemanticCache:
    """Meaning-based LLM response cache backed by Redis + local embeddings."""

    def __init__(
        self,
        redis_client: aioredis.Redis,
        *,
        ttl_seconds: float = 300.0,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self._redis = redis_client
        self._ttl = int(ttl_seconds)
        self._threshold = threshold
        # Load the embedding model once at construction (it's ~80MB, reused)
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def _embed(self, text: str) -> np.ndarray:
        # SentenceTransformer.encode is typed as Any; cast to satisfy mypy strict.
        return np.asarray(self._model.encode(text), dtype=np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    async def get(self, *, model: str, prompt: str) -> RoutedLLMResponse | None:
        """Scan cached entries for one semantically close to `prompt`.

        Returns the cached response if max similarity > threshold, else None.
        """
        from llm.router import RoutedLLMResponse  # avoid circular import

        query_emb = self._embed(prompt)

        # Get all entry ids for this model. With decode_responses=False (our
        # default), smembers returns set[bytes] — decode each id before
        # formatting it into the entry key, otherwise the f-string produces
        # `b'...'` literals and every lookup silently misses.
        entry_ids = await self._redis.smembers(_SEMANTIC_INDEX_KEY)
        if not entry_ids:
            return None

        best_sim = 0.0
        best_payload = None
        for raw_id in entry_ids:
            entry_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            raw = await self._redis.get(f"{_SEMANTIC_ENTRY_PREFIX}{entry_id}")
            if raw is None:
                # Entry expired but id still in index — clean it up lazily
                await self._redis.srem(_SEMANTIC_INDEX_KEY, raw_id)
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # Only compare same-model entries
            if entry.get("model") != model:
                continue
            stored_emb = np.array(entry["embedding"], dtype=np.float32)
            sim = self._cosine(query_emb, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_payload = entry

        if best_payload is not None and best_sim >= self._threshold:
            logger.info(
                "semantic_cache_hit",
                similarity=round(best_sim, 4),
                threshold=self._threshold,
                cached_prompt=best_payload["prompt"][:80],
                query_prompt=prompt[:80],
            )
            return RoutedLLMResponse(**best_payload["response"])

        logger.debug(
            "semantic_cache_miss",
            best_similarity=round(best_sim, 4),
            threshold=self._threshold,
        )
        return None

    async def set(self, *, model: str, prompt: str, response: RoutedLLMResponse) -> None:
        """Store an entry: embedding + prompt + response."""
        import hashlib

        query_emb = self._embed(prompt)
        entry_id = hashlib.sha256(f"{model}|{prompt}".encode()).hexdigest()[:16]

        entry = {
            "model": model,
            "prompt": prompt,
            "embedding": query_emb.tolist(),  # JSON-serializable
            "response": asdict(response),
        }
        # Store entry with TTL, add id to the index set
        await self._redis.set(
            f"{_SEMANTIC_ENTRY_PREFIX}{entry_id}",
            json.dumps(entry, default=str),
            ex=self._ttl,
        )
        await self._redis.sadd(_SEMANTIC_INDEX_KEY, entry_id)
        # The index set itself should expire eventually too, but entries
        # self-clean via the lazy removal in get(). Acceptable for now.
