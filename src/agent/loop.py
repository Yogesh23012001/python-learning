"""Agent loop — yields events as they happen.

The loop is an async generator. Each round of "ask LLM, run tools" emits
events in order: iteration_started, tool_requested(s), tool_completed(s) or
tool_failed(s), then either agent_completed or agent_stopped at the end.

The non-streaming caller can consume this and aggregate; the streaming
caller forwards events to SSE.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from api.config import get_settings
from google import genai
from google.genai import types
from llm.pricing import calculate_cost_usd

from agent.events import (
    AgentCompleted,
    AgentEvent,
    AgentStarted,
    AgentStopped,
    IterationStarted,
    ToolCompleted,
    ToolFailed,
    ToolRequested,
)
from agent.tools import (
    TOOL_SCHEMAS,
    ToolContext,
    ToolRefusedError,
    ToolTimeoutError,
    execute_tool,
)

logger = structlog.get_logger(__name__)


# ============================================================
# Cost helper
# ============================================================


def _safe_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    try:
        return calculate_cost_usd(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens
        )
    except KeyError:
        logger.warning("cost_unknown_model", model=model)
        return 0.0


# ============================================================
# The agent loop — now an async generator
# ============================================================


async def run_agent_stream(
    user_prompt: str,
    *,
    tool_context: ToolContext,
    max_iterations: int = 8,
    max_cost_usd: float = 0.10,
    model: str | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run the agent and yield events as they happen.

    This is the canonical implementation. Both /agent/run (non-streaming)
    and /agent/run/stream (SSE) call this — non-streaming aggregates,
    streaming forwards.
    """
    settings = get_settings()
    if settings.gemini_api_key is None:
        raise RuntimeError("GEMINI_API_KEY not set")
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    chosen_model = model or "gemini-2.5-flash"

    gemini_tools: list[types.Tool | Any] = [
        types.Tool(
            function_declarations=[types.FunctionDeclaration(**schema) for schema in TOOL_SCHEMAS]
        )
    ]

    conversation: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
    ]

    total_input_tokens = 0
    total_output_tokens = 0

    logger.info(
        "agent_stream_started",
        prompt_length=len(user_prompt),
        max_iterations=max_iterations,
        model=chosen_model,
    )
    yield AgentStarted(
        prompt_length=len(user_prompt),
        max_iterations=max_iterations,
        model=chosen_model,
    )

    for iteration in range(1, max_iterations + 1):
        yield IterationStarted(iteration=iteration)

        # ---- Call the LLM ----
        response = await client.aio.models.generate_content(
            model=chosen_model,
            contents=conversation,
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                max_output_tokens=1024,
            ),
        )

        usage = response.usage_metadata
        if usage:
            total_input_tokens += usage.prompt_token_count or 0
            total_output_tokens += usage.candidates_token_count or 0

        # ---- Check cost cap (after each LLM call) ----
        current_cost = _safe_cost(chosen_model, total_input_tokens, total_output_tokens)
        if current_cost > max_cost_usd:
            logger.warning(
                "agent_cost_cap_reached",
                cost_usd=round(current_cost, 6),
                cap_usd=max_cost_usd,
                iterations=iteration,
            )
            yield AgentStopped(
                reason="cost_cap",
                iterations=iteration,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_cost_usd=current_cost,
            )
            return

        # ---- Process response ----
        if not response.candidates:
            raise RuntimeError("model returned no candidates")
        candidate = response.candidates[0]
        if candidate.content is None:
            raise RuntimeError("candidate had no content")
        content = candidate.content
        parts = content.parts or []
        function_calls = [p.function_call for p in parts if p.function_call is not None]
        text_parts = [p.text for p in parts if p.text]

        if not function_calls:
            final_text = "".join(text_parts)
            logger.info(
                "agent_stream_completed",
                iterations=iteration,
                response_length=len(final_text),
                total_cost_usd=round(current_cost, 6),
            )
            yield AgentCompleted(
                text=final_text,
                iterations=iteration,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_cost_usd=current_cost,
            )
            return

        # ---- LLM requested tools — execute them ----
        conversation.append(content)
        tool_response_parts: list[types.Part] = []

        for fc in function_calls:
            if fc.name is None:
                continue
            name: str = fc.name
            args = dict(fc.args) if fc.args else {}

            yield ToolRequested(name=name, args=args, iteration=iteration)

            tool_start = time.perf_counter()
            try:
                result = await execute_tool(name, args, tool_context)
                duration_ms = (time.perf_counter() - tool_start) * 1000
                yield ToolCompleted(
                    name=name,
                    iteration=iteration,
                    result_keys=list(result.keys()),
                    duration_ms=duration_ms,
                )
            except ToolTimeoutError as exc:
                duration_ms = (time.perf_counter() - tool_start) * 1000
                result = {"error": "timeout", "tool": name, "detail": str(exc)}
                yield ToolFailed(
                    name=name, iteration=iteration, error=str(exc), error_type="timeout"
                )
            except ToolRefusedError as exc:
                result = {"error": "tool_refused", "tool": name, "detail": str(exc)}
                yield ToolFailed(
                    name=name, iteration=iteration, error=str(exc), error_type="refused"
                )
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
                yield ToolFailed(
                    name=name,
                    iteration=iteration,
                    error=str(exc)[:200],
                    error_type="exception",
                )

            tool_response_parts.append(
                types.Part.from_function_response(name=name, response=result)
            )

        conversation.append(types.Content(role="user", parts=tool_response_parts))

    # ---- Loop exhausted ----
    final_cost = _safe_cost(chosen_model, total_input_tokens, total_output_tokens)
    logger.warning(
        "agent_max_iterations_reached",
        max_iterations=max_iterations,
        total_cost_usd=round(final_cost, 6),
    )
    yield AgentStopped(
        reason="max_iterations",
        iterations=max_iterations,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_usd=final_cost,
    )
