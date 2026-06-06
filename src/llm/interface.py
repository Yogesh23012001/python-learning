"""Provider-agnostic LLM types and interface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

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
# Agent-round types (provider-neutral conversation + tool calling)
# ============================================================


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the model.

    `id` is required for round-trips: providers like OpenAI/Ollama use it to
    match a tool response back to a specific call. Gemini doesn't expose IDs,
    so the Gemini adapter synthesises one (e.g. `f"call_{name}_{idx}"`).
    """

    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class AgentMessage:
    """One turn in an agent conversation, provider-neutral.

    role:
      - "user"      — the human (or initial prompt)
      - "assistant" — the model's reply (text + optional tool_calls)
      - "tool"      — the result of a tool call (matches an earlier call via tool_call_id)
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # only set on role="tool"
    tool_name: str | None = None  # only set on role="tool" (Gemini needs the name back)


@dataclass(frozen=True)
class AgentRoundResponse:
    """The model's reply for one round of an agent loop.

    Exactly one of `text` / `tool_calls` will typically be populated:
    - text only          → model is done answering
    - tool_calls only    → model needs tool execution
    - both populated     → some providers (Gemini) emit reasoning text alongside tool calls
    """

    text: str
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int


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

    async def agent_round(
        self,
        *,
        model: str,
        messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
        max_output_tokens: int | None = None,
    ) -> AgentRoundResponse:
        """One turn of an agent loop: send the conversation + tools, get reply.

        The conversation history is provider-neutral (`AgentMessage` list);
        adapters translate to/from native formats. Tool schemas are the same
        Gemini-style JSON dicts your TOOL_SCHEMAS already uses; adapters
        rewrap them into OpenAI's `{"type":"function", "function": {...}}`
        shape if needed.
        """
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
