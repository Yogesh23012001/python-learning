"""Provider-agnostic wrapper adding retries, cost tracking, logging, metrics."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any

import redis.asyncio as aioredis
import structlog
from prometheus_client import Counter
from pydantic import BaseModel
from redis.exceptions import RedisError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm.errors import LLMRetryableError
from llm.interface import (
    AgentMessage,
    AgentRoundResponse,
    LLMProvider,
    LLMResponse,
)
from llm.metrics import llm_calls_total
from llm.pricing import calculate_cost_usd
from llm.semantic_cache import SemanticCache

logger = structlog.get_logger(__name__)

llm_cache_total = Counter(
    "llm_cache_total",
    "LLM response cache events",
    labelnames=("outcome",),  # "hit" or "miss"
)


# ============================================================
# Routed response (router-augmented; includes cost and timing)
# ============================================================


@dataclass(frozen=True)
class RoutedLLMResponse:
    """LLMResponse plus router-added metadata."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    cost_usd: float
    elapsed_ms: float


# ============================================================
# The router
# ============================================================


class LLMRouter:
    """Provider-agnostic LLM client with operational concerns added.

    Wraps a single provider (today). Multi-provider routing (Gemini for
    cheap classification, Claude for complex reasoning) is a future
    extension — change the constructor to take many providers and add a
    routing decision in generate().
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        default_model: str,
        redis_client: aioredis.Redis,
        max_retries: int = 3,
        cache_ttl_seconds: float = 300.0,
        enable_semantic_cache: bool = True,
    ) -> None:
        self._provider = provider
        self._default_model = default_model
        self._max_retries = max_retries
        self._cache = _PromptCache(redis_client, ttl_seconds=cache_ttl_seconds)
        # SemanticCache loads an ~80MB sentence-transformer at construction;
        # disable in tests / mock contexts to avoid the cost.
        self._semantic_cache: SemanticCache | None = (
            SemanticCache(redis_client, ttl_seconds=cache_ttl_seconds)
            if enable_semantic_cache
            else None
        )

    @property
    def default_model(self) -> str:
        """The fallback model used when callers don't pass one explicitly."""
        return self._default_model

    @property
    def provider_name(self) -> str:
        """The configured provider's name (e.g. 'gemini', 'anthropic')."""
        return self._provider.name

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        response_schema: type[BaseModel] | None = None,
        max_output_tokens: int | None = None,
    ) -> RoutedLLMResponse:
        chosen_model = model or self._default_model

        # ---- Cache lookup (skip for structured-output calls; see note below) ----
        # Two-tier: exact byte-match first (fastest), then semantic similarity
        # (slower, catches paraphrases). Both degrade to cache miss on Redis
        # errors so the LLM call path is never blocked by cache infra.
        if response_schema is None:
            cached = await self._cache.get(
                model=chosen_model, prompt=prompt, max_tokens=max_output_tokens
            )
            if cached is not None:
                llm_cache_total.labels(outcome="exact_hit").inc()
                logger.info(
                    "llm_cache_hit",
                    cache_type="exact",
                    provider=self._provider.name,
                    model=chosen_model,
                    input_tokens=cached.input_tokens,
                    output_tokens=cached.output_tokens,
                    cost_saved_usd=round(cached.cost_usd, 6),
                )
                return RoutedLLMResponse(
                    text=cached.text,
                    input_tokens=cached.input_tokens,
                    output_tokens=cached.output_tokens,
                    model=cached.model,
                    provider=cached.provider,
                    cost_usd=0.0,
                    elapsed_ms=0.0,
                )

            # Exact miss — try semantic cache for paraphrase-level matches.
            if self._semantic_cache is not None:
                try:
                    sem_cached = await self._semantic_cache.get(model=chosen_model, prompt=prompt)
                except RedisError as exc:
                    logger.warning(
                        "semantic_cache_get_redis_unavailable",
                        error=str(exc)[:100],
                    )
                    sem_cached = None
                if sem_cached is not None:
                    llm_cache_total.labels(outcome="semantic_hit").inc()
                    logger.info(
                        "llm_cache_hit",
                        cache_type="semantic",
                        provider=self._provider.name,
                        model=chosen_model,
                        input_tokens=sem_cached.input_tokens,
                        output_tokens=sem_cached.output_tokens,
                        cost_saved_usd=round(sem_cached.cost_usd, 6),
                    )
                    return RoutedLLMResponse(
                        text=sem_cached.text,
                        input_tokens=sem_cached.input_tokens,
                        output_tokens=sem_cached.output_tokens,
                        model=sem_cached.model,
                        provider=sem_cached.provider,
                        cost_usd=0.0,
                        elapsed_ms=0.0,
                    )
            llm_cache_total.labels(outcome="miss").inc()

        # ---- Existing retry + provider call logic ----
        start = time.perf_counter()

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(LLMRetryableError),
            reraise=True,
        ):
            with attempt:
                try:
                    response: LLMResponse = await self._provider.generate(
                        prompt,
                        model=chosen_model,
                        response_schema=response_schema,
                        max_output_tokens=max_output_tokens,
                    )
                except Exception as exc:
                    llm_calls_total.labels(
                        provider=self._provider.name,
                        model=chosen_model,
                        outcome="failed",
                    ).inc()
                    logger.warning(
                        "llm_call_failed",
                        provider=self._provider.name,
                        model=chosen_model,
                        error_type=type(exc).__name__,
                        error=str(exc)[:200],
                    )
                    raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        cost = _safe_cost(chosen_model, response.input_tokens, response.output_tokens)

        llm_calls_total.labels(
            provider=self._provider.name,
            model=chosen_model,
            outcome="success",
        ).inc()
        logger.info(
            "llm_call_completed",
            provider=self._provider.name,
            model=chosen_model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=round(cost, 6),
            elapsed_ms=round(elapsed_ms, 2),
        )

        result = RoutedLLMResponse(
            text=response.text,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model=chosen_model,
            provider=self._provider.name,
            cost_usd=cost,
            elapsed_ms=elapsed_ms,
        )

        # ---- Cache write (exact + semantic) ----
        if response_schema is None:
            await self._cache.set(
                model=chosen_model,
                prompt=prompt,
                max_tokens=max_output_tokens,
                response=result,
            )
            if self._semantic_cache is not None:
                try:
                    await self._semantic_cache.set(
                        model=chosen_model, prompt=prompt, response=result
                    )
                except RedisError as exc:
                    logger.warning(
                        "semantic_cache_set_redis_unavailable",
                        error=str(exc)[:100],
                    )

        return result

    async def agent_round(
        self,
        *,
        messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> AgentRoundResponse:
        """One round of an agent loop, provider-neutral.

        Records llm_calls_total per attempt (success/failed). The router does
        NOT cache or retry agent rounds — agent loops carry conversation state
        and a cached/retried turn could produce inconsistent histories.
        """
        chosen_model = model or self._default_model
        try:
            response = await self._provider.agent_round(
                model=chosen_model,
                messages=messages,
                tool_schemas=tool_schemas,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            llm_calls_total.labels(
                provider=self._provider.name,
                model=chosen_model,
                outcome="failed",
            ).inc()
            logger.warning(
                "llm_agent_round_failed",
                provider=self._provider.name,
                model=chosen_model,
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            raise

        llm_calls_total.labels(
            provider=self._provider.name,
            model=chosen_model,
            outcome="success",
        ).inc()
        logger.info(
            "llm_agent_round_completed",
            provider=self._provider.name,
            model=chosen_model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            tool_calls=len(response.tool_calls),
        )
        return response

    async def generate_stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield text chunks (router unwraps StreamChunk; caller doesn't see it)."""
        chosen_model = model or self._default_model
        start = time.perf_counter()
        input_tokens = 0
        output_tokens = 0
        try:
            async for chunk in self._provider.generate_stream(
                prompt,
                model=chosen_model,
                max_output_tokens=max_output_tokens,
            ):
                if chunk.is_final:
                    input_tokens = chunk.input_tokens
                    output_tokens = chunk.output_tokens
                elif chunk.text:
                    yield chunk.text
        except Exception as exc:
            llm_calls_total.labels(
                provider=self._provider.name,
                model=chosen_model,
                outcome="failed",
            ).inc()
            logger.warning(
                "llm_stream_failed",
                provider=self._provider.name,
                model=chosen_model,
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        cost = _safe_cost(chosen_model, input_tokens, output_tokens)
        llm_calls_total.labels(
            provider=self._provider.name,
            model=chosen_model,
            outcome="success",
        ).inc()
        logger.info(
            "llm_stream_completed",
            provider=self._provider.name,
            model=chosen_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
            elapsed_ms=round(elapsed_ms, 2),
        )


def _safe_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    try:
        return calculate_cost_usd(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens
        )
    except KeyError:
        logger.warning("unknown_model_pricing", model=model)
        return 0.0


class _PromptCache:
    """Exact-match LLM response cache backed by Redis.

    Keys are sha256(model + max_tokens + prompt). Values are JSON-serialized
    RoutedLLMResponse fields. TTL bounds staleness.

    Migrated from in-memory dict -> Redis so the cache survives restarts and
    is shared across processes/workers.

    Redis errors (connection refused, timeout, etc.) degrade to cache miss —
    we never want a cache outage to break the LLM call path.
    """

    def __init__(self, redis_client: aioredis.Redis, *, ttl_seconds: float = 300.0) -> None:
        self._redis = redis_client
        self._ttl = int(ttl_seconds)

    @staticmethod
    def _key(model: str, prompt: str, max_tokens: int | None) -> str:
        material = f"{model}|{max_tokens}|{prompt}"
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return f"llmcache:exact:{digest}"

    async def get(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int | None,
    ) -> RoutedLLMResponse | None:
        key = self._key(model, prompt, max_tokens)
        try:
            raw = await self._redis.get(key)
        except RedisError as exc:
            logger.warning("cache_get_redis_unavailable", error=str(exc)[:100])
            return None
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return RoutedLLMResponse(**data)
        except (json.JSONDecodeError, TypeError) as exc:
            # Corrupt cache entry — log and treat as miss
            logger.warning("cache_deserialize_failed", key=key, error=str(exc)[:100])
            return None

    async def set(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int | None,
        response: RoutedLLMResponse,
    ) -> None:
        key = self._key(model, prompt, max_tokens)
        payload = json.dumps(asdict(response), default=str)
        try:
            await self._redis.set(key, payload, ex=self._ttl)
        except RedisError as exc:
            logger.warning("cache_set_redis_unavailable", error=str(exc)[:100])
