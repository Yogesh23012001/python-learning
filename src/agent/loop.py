"""Agent loop — yields events as they happen.

The loop is an async generator. Each round of "ask LLM, run tools" emits
events in order: iteration_started, tool_requested(s), tool_completed(s) or
tool_failed(s), then either agent_completed or agent_stopped at the end.

The non-streaming caller can consume this and aggregate; the streaming
caller forwards events to SSE.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import structlog
from llm.interface import AgentMessage
from llm.pricing import calculate_cost_usd
from llm.router import LLMRouter

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
    llm: LLMRouter,
    tool_context: ToolContext,
    max_iterations: int = 8,
    max_cost_usd: float = 0.10,
    model: str | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run the agent and yield events as they happen.

    Provider-agnostic: works with any provider behind LLMRouter (Gemini,
    OpenRouter, local Ollama, mock). The conversation history is kept as
    neutral `AgentMessage` objects; LLMRouter.agent_round translates per
    provider on each turn.
    """
    chosen_model = model or llm.default_model

    conversation: list[AgentMessage] = [
        AgentMessage(role="user", content=user_prompt),
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

        # ---- Call the LLM (provider-neutral) ----
        round_response = await llm.agent_round(
            messages=conversation,
            tool_schemas=TOOL_SCHEMAS,
            model=chosen_model,
            max_output_tokens=1024,
        )

        total_input_tokens += round_response.input_tokens
        total_output_tokens += round_response.output_tokens

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

        # ---- Done when the model produced text and no tool calls ----
        if not round_response.tool_calls:
            logger.info(
                "agent_stream_completed",
                iterations=iteration,
                response_length=len(round_response.text),
                total_cost_usd=round(current_cost, 6),
            )
            yield AgentCompleted(
                text=round_response.text,
                iterations=iteration,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_cost_usd=current_cost,
            )
            return

        # ---- LLM requested tools — record the assistant turn, then execute ----
        conversation.append(
            AgentMessage(
                role="assistant",
                content=round_response.text,
                tool_calls=list(round_response.tool_calls),
            )
        )

        for tc in round_response.tool_calls:
            yield ToolRequested(name=tc.name, args=tc.args, iteration=iteration)

            tool_start = time.perf_counter()
            try:
                result: dict[str, object] = await execute_tool(tc.name, tc.args, tool_context)
                duration_ms = (time.perf_counter() - tool_start) * 1000
                yield ToolCompleted(
                    name=tc.name,
                    iteration=iteration,
                    result_keys=list(result.keys()),
                    duration_ms=duration_ms,
                )
            except ToolTimeoutError as exc:
                result = {"error": "timeout", "tool": tc.name, "detail": str(exc)}
                yield ToolFailed(
                    name=tc.name, iteration=iteration, error=str(exc), error_type="timeout"
                )
            except ToolRefusedError as exc:
                result = {"error": "tool_refused", "tool": tc.name, "detail": str(exc)}
                yield ToolFailed(
                    name=tc.name, iteration=iteration, error=str(exc), error_type="refused"
                )
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
                yield ToolFailed(
                    name=tc.name,
                    iteration=iteration,
                    error=str(exc)[:200],
                    error_type="exception",
                )

            # Append the tool result as the next conversation turn
            conversation.append(
                AgentMessage(
                    role="tool",
                    content=json.dumps(result, default=str),
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                )
            )

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
