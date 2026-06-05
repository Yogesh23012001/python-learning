"""HTTP routes for the agent system — streaming via async generator."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict
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

from agent.events import (
    AgentCompleted,
    AgentEvent,
    AgentStopped,
    ToolCompleted,
    ToolFailed,
    ToolRequested,
)
from agent.loop import run_agent_stream
from agent.tools import ToolContext

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


def get_tool_context(request: Request) -> ToolContext:
    return ToolContext(
        github_client=getattr(request.app.state, "github_client", None),
        session_factory=getattr(request.app.state, "session_factory", None),
    )


ToolContextDep = Annotated[ToolContext, Depends(get_tool_context)]


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
# Non-streaming endpoint — consume the generator, aggregate
# ============================================================


@router.post("/run", response_model=AgentRunResponse)
async def agent_run(
    context: ToolContextDep,
    payload: AgentRunRequest,
) -> AgentRunResponse:
    """Run the agent and return the final aggregated result as JSON."""
    logger.info("agent_run_request", prompt_length=len(payload.prompt))

    tool_calls: list[ToolCallRecord] = []
    terminal: AgentCompleted | AgentStopped | None = None

    try:
        async for event in run_agent_stream(
            payload.prompt,
            tool_context=context,
            max_iterations=payload.max_iterations,
            max_cost_usd=payload.max_cost_usd,
            model=payload.model,
        ):
            if isinstance(event, ToolRequested):
                tool_calls.append(ToolCallRecord(name=event.name, args=event.args))
            elif isinstance(event, ToolCompleted):
                agent_tool_calls_total.labels(tool=event.name, outcome="success").inc()
            elif isinstance(event, ToolFailed):
                agent_tool_calls_total.labels(tool=event.name, outcome=event.error_type).inc()
            elif isinstance(event, AgentCompleted | AgentStopped):
                terminal = event
    except Exception as exc:
        logger.exception("agent_run_failed")
        raise HTTPException(status_code=500, detail=f"agent failed: {exc}") from exc

    if terminal is None:
        # Generator ended without a terminal event — shouldn't happen
        raise HTTPException(status_code=500, detail="agent ended without terminal event")

    # Metrics
    agent_iterations.observe(terminal.iterations)
    agent_cost_usd.observe(terminal.total_cost_usd)

    if isinstance(terminal, AgentStopped):
        agent_runs_total.labels(outcome=terminal.reason).inc()
        raise HTTPException(
            status_code=422,
            detail={
                "error": terminal.reason,
                "message": f"agent stopped due to {terminal.reason}",
                "iterations": terminal.iterations,
                "tool_calls": [tc.model_dump() for tc in tool_calls],
                "total_cost_usd": round(terminal.total_cost_usd, 6),
            },
        )

    # AgentCompleted normal path
    agent_runs_total.labels(outcome="completed").inc()
    return AgentRunResponse(
        text=terminal.text,
        iterations=terminal.iterations,
        tool_calls=tool_calls,
        hit_iteration_limit=False,
        total_input_tokens=terminal.total_input_tokens,
        total_output_tokens=terminal.total_output_tokens,
        total_cost_usd=round(terminal.total_cost_usd, 6),
    )


# ============================================================
# Streaming endpoint — forward events as SSE
# ============================================================


def _event_to_sse(event: AgentEvent) -> str:
    """Convert an event dataclass to an SSE-formatted line."""
    event_type = event.type
    # Exclude the 'type' field from data since it's the SSE event name
    data = {k: v for k, v in asdict(event).items() if k != "type"}
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/run/stream")
async def agent_run_stream(
    context: ToolContextDep,
    payload: AgentRunRequest,
) -> StreamingResponse:
    """Stream agent events live as SSE."""
    logger.info("agent_stream_request", prompt_length=len(payload.prompt))

    async def _generate() -> AsyncIterator[str]:
        try:
            async for event in run_agent_stream(
                payload.prompt,
                tool_context=context,
                max_iterations=payload.max_iterations,
                max_cost_usd=payload.max_cost_usd,
                model=payload.model,
            ):
                yield _event_to_sse(event)
        except Exception as exc:
            logger.exception("agent_stream_failed")
            yield f"event: error\ndata: {json.dumps({'error': str(exc)[:300]})}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
