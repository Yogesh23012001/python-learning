"""Provider-agnostic wrapper adding retries, cost tracking, logging, metrics."""

from __future__ import annotations

import hashlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import structlog
from prometheus_client import Counter
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm.errors import LLMRetryableError
from llm.interface import LLMProvider, LLMResponse
from llm.metrics import llm_calls_total
from llm.pricing import calculate_cost_usd

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
        max_retries: int = 3,
        cache_ttl_seconds: float = 300.0,
    ) -> None:
        self._provider = provider
        self._default_model = default_model
        self._max_retries = max_retries
        self._cache = _PromptCache(ttl_seconds=cache_ttl_seconds)

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
        if response_schema is None:
            cached = self._cache.get(
                model=chosen_model,
                prompt=prompt,
                max_tokens=max_output_tokens,
            )
            if cached is not None:
                llm_cache_total.labels(outcome="hit").inc()
                logger.info(
                    "llm_cache_hit",
                    provider=self._provider.name,
                    model=chosen_model,
                    input_tokens=cached.input_tokens,
                    output_tokens=cached.output_tokens,
                    cost_saved_usd=round(cached.cost_usd, 6),
                )
                # Return cached response — cost and elapsed are now 0
                return RoutedLLMResponse(
                    text=cached.text,
                    input_tokens=cached.input_tokens,
                    output_tokens=cached.output_tokens,
                    model=cached.model,
                    provider=cached.provider,
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

        # ---- Cache write ----
        if response_schema is None:
            self._cache.set(
                model=chosen_model,
                prompt=prompt,
                max_tokens=max_output_tokens,
                response=result,
            )

        return result

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
    """In-memory TTL cache for LLM responses.

    Production replacement: Redis with TTL + LRU eviction.
    This in-memory version is fine for single-process learning + dev.
    """

    def __init__(self, *, ttl_seconds: float = 300.0, max_entries: int = 1000) -> None:
        self._store: dict[str, tuple[float, RoutedLLMResponse]] = {}
        self._ttl = ttl_seconds
        self._max = max_entries

    @staticmethod
    def _key(model: str, prompt: str, max_tokens: int | None) -> str:
        material = f"{model}|{max_tokens}|{prompt}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int | None,
    ) -> RoutedLLMResponse | None:
        key = self._key(model, prompt, max_tokens)
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, response = entry
        if time.time() - ts > self._ttl:
            self._store.pop(key, None)
            return None
        return response

    def set(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int | None,
        response: RoutedLLMResponse,
    ) -> None:
        if len(self._store) >= self._max:
            # Evict the oldest
            oldest = min(self._store.keys(), key=lambda k: self._store[k][0])
            self._store.pop(oldest, None)
        key = self._key(model, prompt, max_tokens)
        self._store[key] = (time.time(), response)
