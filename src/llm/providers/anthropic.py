"""Anthropic Claude provider adapter.

Implements the LLMProvider protocol for Anthropic's Claude models, including
optional prompt caching. Caching is configured at construction time and applies
to the system prompt + tool definitions on every agent_round call.

Cache semantics (Anthropic-specific):
  - cache_control: {"type": "ephemeral"} marks a content block to be cached.
  - Cache TTL defaults to ~5 minutes.
  - Cache write costs 1.25x input tokens; cache read costs 0.1x input tokens.
  - Cache hits are surfaced in response.usage.cache_read_input_tokens.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic
from anthropic._exceptions import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)
from pydantic import BaseModel

from llm.errors import (
    LLMContentBlockedError,
    LLMContextTooLongError,
    LLMEmptyResponseError,
    LLMError,
    LLMOverloadedError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMRetryableError,
)
from llm.interface import (
    AgentMessage,
    AgentRoundResponse,
    LLMResponse,
    StreamChunk,
    ToolCall,
)


def _classify_anthropic_error(exc: Exception) -> LLMError:
    """Map Anthropic SDK errors to our typed hierarchy.

    This is the ONLY place that knows about Anthropic's error shapes.
    """
    if isinstance(exc, APIConnectionError | APITimeoutError):
        return LLMRetryableError(_short(exc))

    if isinstance(exc, APIStatusError):
        status = exc.status_code
        message = str(exc).lower()
        if status == 429:
            if "quota" in message or "credit balance" in message:
                return LLMQuotaExceededError(_short(exc))
            return LLMRateLimitError(_short(exc))
        if status == 400 and ("context" in message or "token" in message or "too long" in message):
            return LLMContextTooLongError(_short(exc))
        if status in (502, 503, 504, 529):  # 529 = Anthropic-specific overload signal
            return LLMOverloadedError(_short(exc))

    return LLMRetryableError(_short(exc))


def _short(exc: Exception) -> str:
    """Trim multi-line error messages for log-friendliness."""
    return str(exc).split("\n")[0][:200]


class AnthropicProvider:
    """Adapter for Anthropic's Claude API.

    Set `enable_prompt_cache=True` to mark the system prompt + tool
    definitions for caching on agent_round. This typically saves 60-70%
    on input cost for multi-iteration agent runs.
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str,
        enable_prompt_cache: bool = False,
    ) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._enable_prompt_cache = enable_prompt_cache

    # ------------------------------------------------------------
    # generate() — non-streaming single call
    # ------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        response_schema: type[BaseModel] | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        if response_schema is not None:
            # Anthropic doesn't have OpenAI-style strict JSON schema mode.
            # The common pattern is to prepend an instruction; we surface
            # the limitation honestly rather than fake support.
            raise NotImplementedError(
                "AnthropicProvider.generate does not support response_schema yet; "
                "wrap the prompt with schema instructions at the caller."
            )

        try:
            response = await self._client.messages.create(
                model=model,
                max_tokens=max_output_tokens or 1024,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise _classify_anthropic_error(exc) from exc

        # Anthropic returns content as a list of blocks (text, tool_use, etc.)
        text_blocks = [b.text for b in response.content if b.type == "text"]
        text = "".join(text_blocks)
        if not text:
            raise LLMEmptyResponseError(f"empty content (stop_reason={response.stop_reason})")
        if response.stop_reason == "stop_sequence" and not text:
            # Defensive: shouldn't happen but easy to misread
            raise LLMContentBlockedError("blocked by stop_sequence")

        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
            provider=self.name,
        )

    # ------------------------------------------------------------
    # generate_stream() — streaming single call
    # ------------------------------------------------------------

    async def generate_stream(
        self,
        prompt: str,
        *,
        model: str,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        try:
            async with self._client.messages.stream(
                model=model,
                max_tokens=max_output_tokens or 1024,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text_delta in stream.text_stream:
                    yield StreamChunk(text=text_delta)

                final = await stream.get_final_message()
        except Exception as exc:
            raise _classify_anthropic_error(exc) from exc

        yield StreamChunk(
            text="",
            is_final=True,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )

    # ------------------------------------------------------------
    # agent_round() — provider-neutral agent loop turn
    # ------------------------------------------------------------

    async def agent_round(
        self,
        *,
        model: str,
        messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
        max_output_tokens: int | None = None,
    ) -> AgentRoundResponse:
        """Translate neutral messages → Anthropic format, run one round."""
        # Extract any system messages — Anthropic takes `system` as a top-level
        # parameter, separate from the user/assistant `messages` array.
        system_blocks, user_messages = _split_system_and_messages(messages)

        # Convert tool schemas to Anthropic's format
        # Anthropic uses {"name", "description", "input_schema"} —
        # input_schema is the JSON schema from your tool definitions.
        anthropic_tools = [
            {
                "name": schema["name"],
                "description": schema["description"],
                "input_schema": schema["parameters"],
            }
            for schema in tool_schemas
        ]

        # Mark the LAST tool definition with cache_control if caching enabled.
        # This caches: system prompt + ALL tool definitions (because anthropic
        # caches up to and including the marked block).
        if self._enable_prompt_cache and anthropic_tools:
            anthropic_tools[-1] = {
                **anthropic_tools[-1],
                "cache_control": {"type": "ephemeral"},
            }

        # If we have a system prompt, also mark it for caching when enabled.
        # Anthropic supports up to 4 cache breakpoints per request.
        if self._enable_prompt_cache and system_blocks:
            system_blocks = [
                {**block, "cache_control": {"type": "ephemeral"}}
                if i == len(system_blocks) - 1  # mark only the last one
                else block
                for i, block in enumerate(system_blocks)
            ]

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_output_tokens or 1024,
            "messages": user_messages,
        }
        if system_blocks:
            kwargs["system"] = system_blocks
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as exc:
            raise _classify_anthropic_error(exc) from exc

        # Extract text and tool_use blocks from response content
        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_chunks.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,  # Anthropic gives us a real ID — no synthesis needed
                        name=block.name,
                        args=dict(block.input) if block.input else {},
                    )
                )

        # Surface cache metrics if caching was used. Anthropic exposes:
        #   - cache_creation_input_tokens: tokens written to cache (cost 1.25x)
        #   - cache_read_input_tokens: tokens loaded from cache (cost 0.1x)
        # We don't currently propagate these into AgentRoundResponse because
        # the neutral type doesn't have a slot for them. Log them so we can
        # verify caching is actually working when we test.
        usage = response.usage
        cache_created = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        if cache_created or cache_read:
            import structlog

            logger = structlog.get_logger(__name__)
            logger.info(
                "anthropic_cache_metrics",
                cache_creation_input_tokens=cache_created,
                cache_read_input_tokens=cache_read,
                input_tokens_uncached=usage.input_tokens,
                model=model,
            )

        return AgentRoundResponse(
            text="".join(text_chunks),
            tool_calls=tool_calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )


# ============================================================
# Helpers — neutral → Anthropic message translation
# ============================================================


def _split_system_and_messages(
    messages: list[AgentMessage],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate system messages from user/assistant/tool messages.

    Anthropic's API takes `system` as a top-level parameter (string or list
    of content blocks), distinct from the `messages` array which only
    contains user/assistant turns.

    Returns (system_blocks, conversation_messages).
    """
    system_blocks: list[dict[str, Any]] = []
    conversation: list[dict[str, Any]] = []

    for m in messages:
        if m.role == "system":
            system_blocks.append({"type": "text", "text": m.content})
        elif m.role == "user":
            conversation.append({"role": "user", "content": m.content})
        elif m.role == "assistant":
            conversation.append(_to_anthropic_assistant(m))
        elif m.role == "tool":
            conversation.append(_to_anthropic_tool_result(m))
        else:
            raise ValueError(f"unknown AgentMessage role: {m.role!r}")

    return system_blocks, conversation


def _to_anthropic_assistant(m: AgentMessage) -> dict[str, Any]:
    """Translate an assistant turn (text + tool_calls) to Anthropic format."""
    content_blocks: list[dict[str, Any]] = []
    if m.content:
        content_blocks.append({"type": "text", "text": m.content})
    for tc in m.tool_calls:
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.args,
            }
        )
    return {"role": "assistant", "content": content_blocks}


def _to_anthropic_tool_result(m: AgentMessage) -> dict[str, Any]:
    """Translate a tool-result turn to Anthropic format.

    Anthropic represents tool results as user-role messages containing
    `tool_result` content blocks, matched to a prior tool_use by `tool_use_id`.
    """
    if m.tool_call_id is None:
        raise ValueError("tool message missing tool_call_id (required by Anthropic)")

    # Anthropic expects content as either a string or a list of blocks.
    # Our neutral type stores tool output as a JSON string in `content`.
    # We pass it through as a string — Anthropic handles either form.
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.content,
            }
        ],
    }
