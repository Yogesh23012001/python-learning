"""HTTP routes for the agent system — streaming via async generator."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from decimal import Decimal
from typing import Annotated, Any

import structlog
from api.config import get_settings
from api.mertics import (
    agent_cost_usd,
    agent_iterations,
    agent_runs_total,
    agent_tool_calls_total,
)
from api.model_orm import AgentRun
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from llm.router import LLMRouter
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

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


def get_llm_router(request: Request) -> LLMRouter:
    """Pull the lifespan-managed LLMRouter from app state."""
    router: LLMRouter | None = getattr(request.app.state, "llm_client", None)
    if router is None:
        raise RuntimeError("llm_client not initialized — check lifespan")
    return router


LLMRouterDep = Annotated[LLMRouter, Depends(get_llm_router)]


def get_tool_context(request: Request) -> ToolContext:
    return ToolContext(
        github_client=getattr(request.app.state, "github_client", None),
        session_factory=getattr(request.app.state, "session_factory", None),
        request_id=getattr(request.state, "request_id", ""),
    )


ToolContextDep = Annotated[ToolContext, Depends(get_tool_context)]

ALLOWED_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "llama3.1:8b",
    # add more as you support them
}


class AgentRunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    # All limits default to Settings values when omitted — see api/config.py:
    # AGENT_MAX_ITERATIONS, AGENT_MAX_COST_USD, AGENT_MAX_OUTPUT_TOKENS.
    max_iterations: int | None = Field(default=None, ge=1, le=15)
    max_cost_usd: float | None = Field(default=None, gt=0.0, le=5.0)
    max_output_tokens: int | None = Field(default=None, ge=64, le=4096)
    model: str | None = Field(default=None)

    @field_validator("prompt")
    @classmethod
    def _reject_obvious_attacks(cls, v: str) -> str:
        # Lightweight defense — not exhaustive, just catches the lazy stuff
        lowered = v.lower()
        if "ignore previous instructions" in lowered or "ignore all prior" in lowered:
            # Not auto-rejecting; logging for monitoring purposes
            # Real defense lives in system prompt + tool denylist
            import structlog

            structlog.get_logger(__name__).warning(
                "potential_prompt_injection",
                prompt_preview=v[:200],
            )
        return v

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v: str | None) -> str | None:
        if v is not None and v not in ALLOWED_MODELS:
            raise ValueError(f"unsupported model: {v!r}. Allowed: {sorted(ALLOWED_MODELS)}")
        return v


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


async def _persist_agent_run(
    *,
    session_factory: async_sessionmaker[AsyncSession] | None,
    request_id: str,
    prompt: str,
    model: str,
    provider: str,
    iterations: int,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    outcome: str,
    text_response: str | None,
    tool_calls: list[dict[str, Any]],
) -> None:
    """Append one row to agent_runs. Swallows errors — audit failure must not
    break the user-visible response.
    """
    if session_factory is None or not request_id:
        return
    try:
        async with session_factory() as session:
            session.add(
                AgentRun(
                    request_id=request_id,
                    prompt=prompt,
                    model=model,
                    provider=provider,
                    iterations=iterations,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=Decimal(str(round(cost_usd, 6))),
                    outcome=outcome,
                    text_response=text_response,
                    tool_calls=tool_calls,
                )
            )
            await session.commit()
    except Exception:
        logger.exception("agent_run_audit_failed", request_id=request_id)


# ============================================================
# Non-streaming endpoint — consume the generator, aggregate
# ============================================================


@router.post("/run", response_model=AgentRunResponse)
async def agent_run(
    context: ToolContextDep,
    llm: LLMRouterDep,
    payload: AgentRunRequest,
) -> AgentRunResponse:
    """Run the agent and return the final aggregated result as JSON."""
    settings = get_settings()
    max_iterations = payload.max_iterations or settings.agent_max_iterations
    max_cost_usd = payload.max_cost_usd or settings.agent_max_cost_usd
    max_output_tokens = payload.max_output_tokens or settings.agent_max_output_tokens

    logger.info(
        "agent_run_request",
        prompt_length=len(payload.prompt),
        max_iterations=max_iterations,
        max_cost_usd=max_cost_usd,
        max_output_tokens=max_output_tokens,
    )

    tool_calls: list[ToolCallRecord] = []
    terminal: AgentCompleted | AgentStopped | None = None
    chosen_model = payload.model or llm.default_model

    try:
        async for event in run_agent_stream(
            payload.prompt,
            llm=llm,
            tool_context=context,
            max_iterations=max_iterations,
            max_cost_usd=max_cost_usd,
            max_output_tokens=max_output_tokens,
            model=payload.model,
            request_id=context.request_id,
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
        await _persist_agent_run(
            session_factory=context.session_factory,
            request_id=context.request_id,
            prompt=payload.prompt,
            model=chosen_model,
            provider=llm.provider_name,
            iterations=0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            outcome="error",
            text_response=str(exc)[:500],
            tool_calls=[tc.model_dump() for tc in tool_calls],
        )
        raise HTTPException(status_code=500, detail=f"agent failed: {exc}") from exc

    if terminal is None:
        # Generator ended without a terminal event — shouldn't happen
        raise HTTPException(status_code=500, detail="agent ended without terminal event")

    # Metrics
    agent_iterations.observe(terminal.iterations)
    agent_cost_usd.observe(terminal.total_cost_usd)

    serialized_tool_calls = [tc.model_dump() for tc in tool_calls]

    if isinstance(terminal, AgentStopped):
        agent_runs_total.labels(outcome=terminal.reason).inc()
        await _persist_agent_run(
            session_factory=context.session_factory,
            request_id=context.request_id,
            prompt=payload.prompt,
            model=chosen_model,
            provider=llm.provider_name,
            iterations=terminal.iterations,
            input_tokens=terminal.total_input_tokens,
            output_tokens=terminal.total_output_tokens,
            cost_usd=terminal.total_cost_usd,
            outcome=terminal.reason,
            text_response=None,
            tool_calls=serialized_tool_calls,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": terminal.reason,
                "message": f"agent stopped due to {terminal.reason}",
                "iterations": terminal.iterations,
                "tool_calls": serialized_tool_calls,
                "total_cost_usd": round(terminal.total_cost_usd, 6),
            },
        )

    # AgentCompleted normal path
    agent_runs_total.labels(outcome="completed").inc()
    await _persist_agent_run(
        session_factory=context.session_factory,
        request_id=context.request_id,
        prompt=payload.prompt,
        model=chosen_model,
        provider=llm.provider_name,
        iterations=terminal.iterations,
        input_tokens=terminal.total_input_tokens,
        output_tokens=terminal.total_output_tokens,
        cost_usd=terminal.total_cost_usd,
        outcome="completed",
        text_response=terminal.text,
        tool_calls=serialized_tool_calls,
    )
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
    llm: LLMRouterDep,
    payload: AgentRunRequest,
) -> StreamingResponse:
    """Stream agent events live as SSE."""
    settings = get_settings()
    max_iterations = payload.max_iterations or settings.agent_max_iterations
    max_cost_usd = payload.max_cost_usd or settings.agent_max_cost_usd
    max_output_tokens = payload.max_output_tokens or settings.agent_max_output_tokens

    logger.info(
        "agent_stream_request",
        prompt_length=len(payload.prompt),
        max_iterations=max_iterations,
        max_cost_usd=max_cost_usd,
        max_output_tokens=max_output_tokens,
    )

    async def _generate() -> AsyncIterator[str]:
        try:
            async for event in run_agent_stream(
                payload.prompt,
                llm=llm,
                tool_context=context,
                max_iterations=max_iterations,
                max_cost_usd=max_cost_usd,
                max_output_tokens=max_output_tokens,
                model=payload.model,
                request_id=context.request_id,
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
