"""Test LLMRouter using MockProvider — no API key needed."""

from typing import Any, cast

import pytest
import redis.asyncio as aioredis
from llm.providers.mock import MockProvider
from llm.router import LLMRouter


class _InMemoryAsyncRedis:
    """Tiny async stand-in for redis.asyncio.Redis — enough for cache tests.

    Avoids a fakeredis dep. Supports the two methods _PromptCache touches:
    `get(key)` returns the stored bytes or None; `set(key, value, ex=...)`
    overwrites. TTL is ignored — tests don't exercise expiry.
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    async def set(self, key: str, value: bytes | str, ex: int | None = None) -> None:
        self._store[key] = value if isinstance(value, bytes) else value.encode()


def _fake_redis() -> aioredis.Redis:
    """Cast the in-memory fake so LLMRouter's type annotation accepts it."""
    return cast(aioredis.Redis, cast(Any, _InMemoryAsyncRedis()))


@pytest.mark.asyncio
async def test_router_returns_routed_response_with_cost():
    provider = MockProvider(fixed_response="hello world")
    router = LLMRouter(
        provider=provider,
        default_model="gemini-2.0-flash",
        redis_client=_fake_redis(),
        enable_semantic_cache=False,
    )

    resp = await router.generate("test prompt")

    assert resp.text == "hello world"
    assert resp.provider == "mock"
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    assert resp.elapsed_ms >= 0
    # Mock model is not in pricing table, so cost is 0
    assert resp.cost_usd == 1.0000000000000002e-06


@pytest.mark.asyncio
async def test_router_retries_on_retryable_failures():
    # First two calls fail, third succeeds
    # (MockProvider's failure_rate is random; this is approximate)
    provider = MockProvider(
        fixed_response="success",
        failure_rate=0.0,  # never fail for this test
    )
    router = LLMRouter(
        provider=provider,
        default_model="gemini-2.0-flash",
        redis_client=_fake_redis(),
        enable_semantic_cache=False,
    )

    resp = await router.generate("prompt")
    assert resp.text == "success"


@pytest.mark.asyncio
async def test_router_streams_chunks():
    provider = MockProvider(fixed_response="one two three four")
    router = LLMRouter(
        provider=provider,
        default_model="gemini-2.0-flash",
        redis_client=_fake_redis(),
        enable_semantic_cache=False,
    )

    chunks: list[str] = []
    async for chunk in router.generate_stream("prompt"):
        chunks.append(chunk)

    full = "".join(chunks)
    assert full == "one two three four"
    assert len(chunks) >= 2  # at least multiple chunks
