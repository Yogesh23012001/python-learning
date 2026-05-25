"""Production-grade LLM client wrapping Gemini.

Adds: error classification, cost tracking, structured logging, retries.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import structlog
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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
from llm.metrics import llm_calls_total
from llm.pricing import calculate_cost_usd

logger = structlog.get_logger(__name__)


# ============================================================
# Public types
# ============================================================


@dataclass(frozen=True)
class LLMResponse:
    """Result of a single completion call."""

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    elapsed_ms: float


# ============================================================
# Error classification
# ============================================================


def _classify_error(exc: Exception) -> LLMError:
    """Convert provider-specific exceptions into our typed hierarchy."""
    # genai-specific
    if isinstance(exc, genai_errors.ClientError):
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        message = str(exc).lower()

        if status == 429:
            if "quota" in message:
                return LLMQuotaExceededError(str(exc))
            return LLMRateLimitError(str(exc))
        if status == 400 and ("context" in message or "token" in message):
            return LLMContextTooLongError(str(exc))
        if status == 503:
            return LLMOverloadedError(str(exc))

    if isinstance(exc, genai_errors.ServerError):
        return LLMOverloadedError(str(exc))

    # Anything else — assume retryable network-style failure
    return LLMRetryableError(str(exc))


# ============================================================
# The client
# ============================================================


class LLMClient:
    """Async wrapper around Gemini with production patterns."""

    def __init__(
        self,
        *,
        api_key: str,
        default_model: str,
        max_retries: int = 3,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._default_model = default_model
        self._max_retries = max_retries

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        response_schema: type[BaseModel] | None = None,
        max_output_tokens: int | None = None,
        thinking: bool = False,
    ) -> LLMResponse:
        """Make a single generation call with retries + cost tracking.

        Gemini 2.5 models think before answering; those reasoning tokens
        count against max_output_tokens. Default thinking=False keeps the
        budget for visible output. Pass thinking=True for tasks that
        actually benefit from chain-of-thought (math, code reasoning).
        """
        chosen_model = model or self._default_model

        config: dict[str, Any] = {}
        if max_output_tokens is not None:
            config["max_output_tokens"] = max_output_tokens
        if response_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_schema
        if not thinking:
            # thinking_budget=0 disables internal reasoning for Gemini 2.5
            # models so max_output_tokens applies to user-visible output only.
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(LLMRetryableError),
            reraise=True,
        ):
            with attempt:
                # outcome defaults to "failed"; only overwritten to "success"
                # when we're about to return a valid LLMResponse. finally
                # guarantees one increment per attempt regardless of exit path.
                outcome = "failed"
                start = time.perf_counter()
                try:
                    try:
                        response = await self._client.aio.models.generate_content(
                            model=chosen_model,
                            contents=prompt,
                            config=types.GenerateContentConfig(**config) if config else None,
                        )
                    except Exception as exc:
                        classified = _classify_error(exc)
                        logger.warning(
                            "llm_call_failed",
                            model=chosen_model,
                            error_type=type(classified).__name__,
                            error=str(classified),
                        )
                        raise classified from exc

                    elapsed_ms = (time.perf_counter() - start) * 1000

                    # Validate response — sometimes API returns no text (safety block, etc.)
                    if not response.candidates:
                        raise LLMEmptyResponseError("model returned no candidates")
                    candidate = response.candidates[0]
                    if candidate.finish_reason == types.FinishReason.SAFETY:
                        raise LLMContentBlockedError(
                            f"safety filter triggered: {candidate.safety_ratings}"
                        )
                    if not response.text:
                        raise LLMEmptyResponseError(
                            f"empty text (finish_reason={candidate.finish_reason})"
                        )

                    # Extract usage
                    usage = response.usage_metadata
                    input_tokens = usage.prompt_token_count or 0 if usage else 0
                    output_tokens = usage.candidates_token_count or 0 if usage else 0

                    # Cost
                    try:
                        cost = calculate_cost_usd(
                            model=chosen_model,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        )
                    except KeyError:
                        logger.warning("unknown_model_pricing", model=chosen_model)
                        cost = 0.0

                    logger.info(
                        "llm_call_completed",
                        model=chosen_model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=round(cost, 6),
                        elapsed_ms=round(elapsed_ms, 2),
                    )

                    outcome = "success"
                    return LLMResponse(
                        text=response.text,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=cost,
                        model=chosen_model,
                        elapsed_ms=elapsed_ms,
                    )
                finally:
                    llm_calls_total.labels(
                        provider="gemini",
                        model=chosen_model,
                        outcome=outcome,
                    ).inc()

        raise RuntimeError("unreachable")

    async def generate_stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream text chunks from the model as they're generated."""
        chosen_model = model or self._default_model
        start = time.perf_counter()

        config_dict: dict[str, Any] = {}
        if max_output_tokens is not None:
            config_dict["max_output_tokens"] = max_output_tokens

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=chosen_model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_dict) if config_dict else None,
            )
        except Exception as exc:
            classified = _classify_error(exc)
            logger.warning(
                "llm_stream_failed",
                model=chosen_model,
                error_type=type(classified).__name__,
            )
            raise classified from exc

        input_tokens = 0
        output_tokens = 0

        async for chunk in stream:
            if chunk.text:
                yield chunk.text
            # The final chunk carries usage metadata
            if (usage := chunk.usage_metadata) is not None:
                input_tokens = usage.prompt_token_count or 0
                output_tokens = usage.candidates_token_count or 0

        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            cost = calculate_cost_usd(
                model=chosen_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        except KeyError:
            cost = 0.0

        logger.info(
            "llm_stream_completed",
            model=chosen_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
            elapsed_ms=round(elapsed_ms, 2),
        )
