"""Agent loop with injected tool context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from api.config import get_settings
from google import genai
from google.genai import types
from llm.pricing import calculate_cost_usd

from agent.tools import TOOL_SCHEMAS, ToolContext, ToolRefusedError, ToolTimeoutError, execute_tool

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AgentResult:
    text: str
    iterations: int
    tool_calls: list[tuple[str, dict[str, Any]]]
    hit_iteration_limit: bool
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0


async def run_agent(
    user_prompt: str,
    *,
    tool_context: ToolContext,
    max_iterations: int = 8,
    max_cost_usd: float = 0.10,
    model: str | None = None,
) -> AgentResult:
    """Run the ReAct agent loop. Resources are injected via `tool_context`."""
    settings = get_settings()
    if settings.gemini_api_key is None:
        raise RuntimeError("GEMINI_API_KEY not set")
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    chosen_model = model or "gemini-2.5-flash-lite"

    gemini_tools: list[types.Tool | Any] = [
        types.Tool(
            function_declarations=[types.FunctionDeclaration(**schema) for schema in TOOL_SCHEMAS]
        )
    ]

    conversation: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
    ]
    tool_calls_log: list[tuple[str, dict[str, Any]]] = []

    logger.info(
        "agent_loop_started",
        prompt=user_prompt[:200],
        max_iterations=max_iterations,
        model=chosen_model,
    )
    total_input_tokens = 0
    total_output_tokens = 0
    for iteration in range(1, max_iterations + 1):
        logger.info("agent_iteration_started", iteration=iteration)

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
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        # ---- Cost cap check: stop early if we've blown the budget ----
        current_cost = _safe_cost(chosen_model, total_input_tokens, total_output_tokens)
        if current_cost > max_cost_usd:
            logger.warning(
                "agent_loop_cost_cap_reached",
                cost_usd=round(current_cost, 6),
                cap_usd=max_cost_usd,
                iterations=iteration,
                total_tool_calls=len(tool_calls_log),
            )
            return AgentResult(
                text="",
                iterations=iteration,
                tool_calls=tool_calls_log,
                hit_iteration_limit=False,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_cost_usd=current_cost,
            )

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
                "agent_loop_completed",
                iterations=iteration,
                total_tool_calls=len(tool_calls_log),
                response_length=len(final_text),
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_cost_usd=round(
                    _safe_cost(chosen_model, total_input_tokens, total_output_tokens), 6
                ),
            )
            return AgentResult(
                text=final_text,
                iterations=iteration,
                tool_calls=tool_calls_log,
                hit_iteration_limit=False,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_cost_usd=_safe_cost(chosen_model, total_input_tokens, total_output_tokens),
            )

        conversation.append(content)

        tool_response_parts: list[types.Part] = []
        for fc in function_calls:
            if fc.name is None:
                continue
            name: str = fc.name
            args = dict(fc.args) if fc.args else {}
            tool_calls_log.append((name, args))

            logger.info("llm_requested_tool", iteration=iteration, tool=name, args=args)

            try:
                result = await execute_tool(name, args, tool_context)
            except ToolTimeoutError as exc:
                result = {"error": "timeout", "tool": name, "detail": str(exc)}
            except ToolRefusedError as exc:
                result = {"error": "refused", "tool": name, "detail": str(exc)}
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}

            tool_response_parts.append(
                types.Part.from_function_response(name=name, response=result)
            )

        conversation.append(types.Content(role="user", parts=tool_response_parts))

    logger.warning(
        "agent_loop_max_iterations_reached",
        max_iterations=max_iterations,
        total_tool_calls=len(tool_calls_log),
    )
    return AgentResult(
        text="",
        iterations=max_iterations,
        tool_calls=tool_calls_log,
        hit_iteration_limit=True,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost_usd=_safe_cost(chosen_model, total_input_tokens, total_output_tokens),
    )


def _safe_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    try:
        return calculate_cost_usd(
            model=model, input_tokens=input_tokens, output_tokens=output_tokens
        )
    except KeyError:
        logger.warning("cost_unknown_model", model=model)
        return 0.0
