"""HTTP routes for LLM interactions."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from llm.errors import (
    LLMContentBlockedError,
    LLMContextTooLongError,
    LLMEmptyResponseError,
    LLMPermanentError,
    LLMQuotaExceededError,
    LLMRetryableError,
)
from llm.router import LLMRouter
from pydantic import BaseModel, Field


def get_llm_client(request: Request) -> LLMRouter:
    client: LLMRouter | None = getattr(request.app.state, "llm_client", None)
    if client is None:
        raise RuntimeError("llm_client not initialized — check lifespan")
    return client


LLMClientDep = Annotated[LLMRouter, Depends(get_llm_client)]
logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/llm", tags=["llm"])


# ============================================================
# Dependency
# ============================================================


# def get_llm_client(request: Request) -> LLMClient:
#     client: LLMClient | None = getattr(request.app.state, "llm_client", None)
#     if client is None:
#         raise RuntimeError("llm_client not initialized — check lifespan")
#     return client


# LLMClientDep = Annotated[LLMClient, Depends(get_llm_client)]


# ============================================================
# Request/response models
# ============================================================


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10_000)
    max_output_tokens: int = Field(default=500, ge=1, le=4000)
    model: str | None = Field(default=None, description="Override default model")


class ChatResponse(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    elapsed_ms: float


# ============================================================
# Non-streaming endpoint (simple JSON)
# ============================================================


@router.post("/chat", response_model=ChatResponse)
async def chat(client: LLMClientDep, payload: ChatRequest) -> ChatResponse:
    """Generate a complete response. Returns JSON when finished."""
    logger.info("chat_request_received", prompt_length=len(payload.prompt))
    try:
        response = await client.generate(
            payload.prompt,
            model=payload.model,
            max_output_tokens=payload.max_output_tokens,
        )
    except LLMContentBlockedError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except LLMContextTooLongError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except LLMQuotaExceededError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    except LLMEmptyResponseError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except LLMRetryableError as e:
        # Even after retries exhausted
        raise HTTPException(status_code=503, detail=str(e)) from e
    except LLMPermanentError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return ChatResponse(
        text=response.text,
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=round(response.cost_usd, 6),
        elapsed_ms=round(response.elapsed_ms, 2),
    )


# ============================================================
# Streaming endpoint (SSE)
# ============================================================


async def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format a single SSE event."""
    payload = json.dumps(data)
    return f"event: {event_type}\ndata: {payload}\n\n"


async def _stream_chat(
    client: LLMRouter,
    payload: ChatRequest,
) -> AsyncIterator[str]:
    """Yield SSE-formatted events as the LLM produces tokens."""
    try:
        async for chunk in client.generate_stream(
            payload.prompt,
            model=payload.model,
            max_output_tokens=payload.max_output_tokens,
        ):
            yield await _sse_event("token", {"text": chunk})
        yield await _sse_event("done", {})
    except LLMQuotaExceededError as e:
        yield await _sse_event("error", {"type": "quota_exceeded", "detail": str(e)[:200]})
    except LLMContentBlockedError as e:
        yield await _sse_event("error", {"type": "content_blocked", "detail": str(e)[:200]})
    except LLMRetryableError as e:
        yield await _sse_event("error", {"type": "upstream_unavailable", "detail": str(e)[:200]})
    except LLMPermanentError as e:
        yield await _sse_event("error", {"type": "permanent_failure", "detail": str(e)[:200]})
    except asyncio.CancelledError:
        logger.info("chat_stream_cancelled_by_client")
        raise


@router.post("/chat/stream")
async def chat_stream(client: LLMClientDep, payload: ChatRequest) -> StreamingResponse:
    """Stream a response token by token via Server-Sent Events."""
    logger.info("chat_stream_request_received", prompt_length=len(payload.prompt))
    return StreamingResponse(
        _stream_chat(client, payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering
            "Connection": "keep-alive",
        },
    )


class Intent(StrEnum):
    QUESTION = "question"
    COMMAND = "command"
    GREETING = "greeting"
    OTHER = "other"


class IntentClassification(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class ClassifyRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


@router.post("/classify-intent", response_model=IntentClassification)
async def classify_intent(
    client: LLMClientDep,
    payload: ClassifyRequest,
) -> IntentClassification:
    """Classify a user message into an intent category."""
    logger.info("intent_classification_requested", message_length=len(payload.message))
    try:
        response = await client.generate(
            f"Classify this user message: {payload.message!r}\nReturn ONLY the structured classification.",
            response_schema=IntentClassification,
        )
    except LLMContentBlockedError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except LLMContextTooLongError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except LLMQuotaExceededError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    except LLMPermanentError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except LLMRetryableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return IntentClassification.model_validate_json(response.text)
