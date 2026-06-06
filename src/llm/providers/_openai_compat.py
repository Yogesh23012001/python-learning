"""Shared OpenAI-compatible helpers used by LocalOllamaProvider and OpenRouterProvider.

Both providers speak the OpenAI Chat Completions protocol. The only differences
are base_url and headers (handled at client construction). Tool-calling
translation is identical, so we keep it here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from llm.errors import LLMEmptyResponseError, LLMError
from llm.interface import AgentMessage, AgentRoundResponse, ToolCall


async def openai_agent_round(
    *,
    client: AsyncOpenAI,
    classify: Callable[[Exception], LLMError],
    model: str,
    messages: list[AgentMessage],
    tool_schemas: list[dict[str, Any]],
    max_output_tokens: int | None = None,
) -> AgentRoundResponse:
    """Run one agent round against any OpenAI Chat Completions endpoint."""
    # OpenAI's SDK uses TypedDicts that are stricter than what we build at
    # runtime; cast at the boundary instead of restructuring every dict.
    openai_messages = cast(
        list[ChatCompletionMessageParam], [_to_openai_message(m) for m in messages]
    )
    openai_tools = cast(
        list[ChatCompletionToolParam],
        [{"type": "function", "function": schema} for schema in tool_schemas],
    )

    extra_kwargs: dict[str, Any] = {"tools": openai_tools}
    if max_output_tokens is not None:
        extra_kwargs["max_tokens"] = max_output_tokens

    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=openai_messages,
            **extra_kwargs,
        )
    except Exception as exc:
        raise classify(exc) from exc

    if not completion.choices:
        raise LLMEmptyResponseError("OpenAI-compat endpoint returned no choices")
    choice = completion.choices[0]
    msg = choice.message

    tool_calls: list[ToolCall] = []
    for tc in msg.tool_calls or []:
        # `tool_calls` can be FunctionToolCall or CustomToolCall in newer SDKs;
        # only the function variant has a `.function` attr. Filter by attribute.
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        try:
            args = json.loads(fn.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_raw_arguments": fn.arguments}
        tool_calls.append(ToolCall(id=tc.id, name=fn.name, args=args))

    usage = completion.usage
    return AgentRoundResponse(
        text=msg.content or "",
        tool_calls=tool_calls,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


def _to_openai_message(m: AgentMessage) -> dict[str, Any]:
    """Translate one neutral AgentMessage into the OpenAI Chat Completions format."""
    if m.role == "user":
        return {"role": "user", "content": m.content}

    if m.role == "assistant":
        msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
        if m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                }
                for tc in m.tool_calls
            ]
        return msg

    if m.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": m.tool_call_id or "",
            "content": m.content,
        }

    raise ValueError(f"unknown AgentMessage role: {m.role!r}")
