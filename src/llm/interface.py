"""Provider-agnostic LLM types and interface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

# ============================================================
# Shared response types
# ============================================================


@dataclass(frozen=True)
class LLMResponse:
    """Result of a single completion call (provider-agnostic)."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    # Cost and elapsed are calculated by the router, not the provider


@dataclass(frozen=True)
class StreamChunk:
    """One chunk from a streaming response."""

    text: str
    is_final: bool = False
    # Final chunk carries usage; intermediate chunks don't
    input_tokens: int = 0
    output_tokens: int = 0


# ============================================================
# The provider interface
# ============================================================


@runtime_checkable
class LLMProvider(Protocol):
    """Contract every provider adapter must implement.

    The router wraps these calls with retries, cost tracking, and logging.
    Providers themselves stay thin and provider-specific.
    """

    name: str
    """Provider name for logs and metrics: 'gemini', 'openai', 'anthropic', 'mock'."""

    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        response_schema: type[BaseModel] | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Make a single non-streaming call."""
        ...

    def generate_stream(
        self,
        prompt: str,
        *,
        model: str,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Yield chunks as the model produces them.

        Declared `def` (not `async def`) returning AsyncIterator so that
        implementations are async generator functions — `async for chunk in
        provider.generate_stream(...)` works directly without an await.
        """
        ...
