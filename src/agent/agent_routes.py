"""HTTP routes for the agent system."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

import structlog
from api.mertics import (
    agent_cost_usd,
    agent_iterations,
    agent_runs_total,
    agent_tool_calls_total,
)
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.loop import AgentResult, run_agent
from agent.tools import ToolContext

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


def _record_agent_metrics(result: AgentResult) -> None:
    """Single source of truth for agent metric increments."""
    agent_runs_total.labels(
        outcome="max_iterations" if result.hit_iteration_limit else "completed"
    ).inc()
    agent_iterations.observe(result.iterations)
    for name, _args in result.tool_calls:
        # We can't tell success vs error from the call log alone in our current
        # shape; mark all as "executed". Hour 5 / Tuesday refactor will let us
        # distinguish success vs error here.
        agent_tool_calls_total.labels(tool=name, outcome="executed").inc()


# ============================================================
# Dependency — assemble the ToolContext from app.state
# ============================================================


def get_tool_context(request: Request) -> ToolContext:
    """Construct a ToolContext from lifespan-managed resources."""
    github_client = getattr(request.app.state, "github_client", None)
    session_factory = getattr(request.app.state, "session_factory", None)
    return ToolContext(
        github_client=github_client,
        session_factory=session_factory,
    )


ToolContextDep = Annotated[ToolContext, Depends(get_tool_context)]


# ============================================================
# Request/response models
# ============================================================


class AgentRunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_iterations: int = Field(default=8, ge=1, le=15)
    max_cost_usd: float = Field(default=0.10, gt=0.0, le=5.0)
    model: str | None = Field(default=None)


class ToolCallRecord(BaseModel):
    name: str
    args: dict[str, Any]


class AgentRunResponse(BaseModel):
    text: str
    iterations: int
    tool_calls: list[ToolCallRecord]
    hit_iteration_limit: bool
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0


# ============================================================
# Non-streaming endpoint
# ============================================================


@router.post("/run", response_model=AgentRunResponse)
async def agent_run(
    context: ToolContextDep,
    payload: AgentRunRequest,
) -> AgentRunResponse:
    logger.info(
        "agent_run_request",
        prompt_length=len(payload.prompt),
        max_iterations=payload.max_iterations,
    )

    try:
        result = await run_agent(
            payload.prompt,
            tool_context=context,
            max_iterations=payload.max_iterations,
            max_cost_usd=payload.max_cost_usd,
            model=payload.model,
        )
    except Exception as exc:
        logger.exception("agent_run_failed")
        raise HTTPException(status_code=500, detail=f"agent failed: {exc}") from exc

    # Update metrics
    if result.hit_iteration_limit:
        agent_runs_total.labels(outcome="max_iterations").inc()
    elif not result.text:
        # Empty text without iteration limit hit = likely cost cap
        agent_runs_total.labels(outcome="cost_cap").inc()
    else:
        agent_runs_total.labels(outcome="completed").inc()

    agent_iterations.observe(result.iterations)
    agent_cost_usd.observe(result.total_cost_usd)
    for name, _args in result.tool_calls:
        agent_tool_calls_total.labels(tool=name, outcome="executed").inc()

    response = AgentRunResponse(
        text=result.text,
        iterations=result.iterations,
        tool_calls=[ToolCallRecord(name=name, args=args) for name, args in result.tool_calls],
        hit_iteration_limit=result.hit_iteration_limit,
        total_input_tokens=result.total_input_tokens,
        total_output_tokens=result.total_output_tokens,
        total_cost_usd=round(result.total_cost_usd, 6),
    )

    # Map safety exits to non-200 status
    if result.hit_iteration_limit:
        # 422 — "I understood your request but I can't fulfill it within limits"
        raise HTTPException(
            status_code=422,
            detail={
                "error": "max_iterations_reached",
                "message": f"agent exceeded max_iterations={payload.max_iterations}",
                "iterations": result.iterations,
                "tool_calls": [{"name": n, "args": a} for n, a in result.tool_calls],
                "total_cost_usd": round(result.total_cost_usd, 6),
            },
        )
    if not result.text:
        # Empty text without iteration cap — likely cost cap or model refused to answer
        raise HTTPException(
            status_code=422,
            detail={
                "error": "no_text_response",
                "message": "agent stopped without producing a final text answer",
                "iterations": result.iterations,
                "tool_calls": [{"name": n, "args": a} for n, a in result.tool_calls],
                "total_cost_usd": round(result.total_cost_usd, 6),
            },
        )

    return response


# ============================================================
# Streaming endpoint — see each step as it happens
# ============================================================


async def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event_type}\ndata: {payload}\n\n"


async def _stream_agent_events(
    payload: AgentRunRequest,
    context: ToolContext,
) -> AsyncIterator[str]:
    """Stream agent events as they happen.

    Implementation note: our `run_agent` returns the final result, not a stream
    of events. For Hour 4 we approximate by running the agent to completion,
    then emitting the events from the result. Hour 5 / Tuesday will refactor
    `run_agent` to yield events as they happen.
    """
    yield await _sse_event("agent_started", {"prompt": payload.prompt[:200]})

    try:
        result = await run_agent(
            payload.prompt,
            tool_context=context,
            max_iterations=payload.max_iterations,
            model=payload.model,
        )
    except Exception as exc:
        yield await _sse_event("error", {"detail": str(exc)[:300]})
        return

    _record_agent_metrics(result)

    # Re-play the captured tool calls as events
    for name, args in result.tool_calls:
        yield await _sse_event("tool_call", {"name": name, "args": args})

    if result.hit_iteration_limit:
        yield await _sse_event(
            "max_iterations_reached",
            {
                "iterations": result.iterations,
                "tool_calls_count": len(result.tool_calls),
            },
        )
    else:
        yield await _sse_event("final_text", {"text": result.text})

    yield await _sse_event(
        "done",
        {
            "iterations": result.iterations,
            "tool_calls": len(result.tool_calls),
        },
    )


@router.post("/run/stream")
async def agent_run_stream(
    context: ToolContextDep,
    payload: AgentRunRequest,
) -> StreamingResponse:
    """Stream the agent's progress as Server-Sent Events."""
    logger.info("agent_stream_request", prompt_length=len(payload.prompt))
    return StreamingResponse(
        _stream_agent_events(payload, context),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
