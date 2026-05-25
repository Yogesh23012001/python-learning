"""Test LLMRouter using MockProvider — no API key needed."""

import pytest
from llm.providers.mock import MockProvider
from llm.router import LLMRouter


@pytest.mark.asyncio
async def test_router_returns_routed_response_with_cost():
    provider = MockProvider(fixed_response="hello world")
    router = LLMRouter(provider=provider, default_model="gemini-2.0-flash")

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
    router = LLMRouter(provider=provider, default_model="gemini-2.0-flash")

    resp = await router.generate("prompt")
    assert resp.text == "success"


@pytest.mark.asyncio
async def test_router_streams_chunks():
    provider = MockProvider(fixed_response="one two three four")
    router = LLMRouter(provider=provider, default_model="gemini-2.0-flash")

    chunks: list[str] = []
    async for chunk in router.generate_stream("prompt"):
        chunks.append(chunk)

    full = "".join(chunks)
    assert full == "one two three four"
    assert len(chunks) >= 2  # at least multiple chunks
