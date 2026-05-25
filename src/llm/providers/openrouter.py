"""OpenRouter provider adapter.

OpenRouter is an aggregator providing OpenAI-compatible access to dozens
of LLM models from many providers (Anthropic, Google, Meta, DeepSeek, etc.).
We talk to it via the OpenAI SDK pointed at OpenRouter's endpoint.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletionUserMessageParam
from pydantic import BaseModel

from llm.errors import (
    LLMContextTooLongError,
    LLMEmptyResponseError,
    LLMError,
    LLMOverloadedError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMRetryableError,
)
from llm.interface import LLMResponse, StreamChunk

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _classify_openrouter_error(exc: Exception) -> LLMError:
    """Convert OpenAI-SDK errors into our shared typed hierarchy."""
    if isinstance(exc, OpenAIRateLimitError):
        # OpenRouter daily-cap errors come through with rate-limit semantics
        # but the message usually clarifies "free-models-per-day" vs minute-rate
        msg = str(exc).lower()
        if "daily" in msg or "per-day" in msg or "quota" in msg:
            return LLMQuotaExceededError(_short(exc))
        return LLMRateLimitError(_short(exc))

    if isinstance(exc, APIStatusError):
        status = exc.status_code
        if status == 429:
            msg = str(exc).lower()
            if "daily" in msg or "quota" in msg:
                return LLMQuotaExceededError(_short(exc))
            return LLMRateLimitError(_short(exc))
        if status == 400 and ("context" in str(exc).lower() or "tokens" in str(exc).lower()):
            return LLMContextTooLongError(_short(exc))
        if status in (502, 503, 504):
            return LLMOverloadedError(_short(exc))

    if isinstance(exc, APIConnectionError | APITimeoutError):
        return LLMRetryableError(_short(exc))

    if isinstance(exc, APIError):
        return LLMRetryableError(_short(exc))

    return LLMRetryableError(_short(exc))


def _short(exc: Exception) -> str:
    return str(exc).split("\n")[0][:200]


class OpenRouterProvider:
    """OpenAI-compatible adapter pointed at OpenRouter."""

    name = "openrouter"

    def __init__(self, *, api_key: str) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                # OpenRouter uses these for usage attribution & ranking
                "HTTP-Referer": "https://github.com/yogesh23012001/python-learning",
                "X-Title": "python-learning",
            },
        )

    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        response_schema: type[BaseModel] | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        messages: list[ChatCompletionUserMessageParam] = [
            {"role": "user", "content": prompt},
        ]

        # OpenAI-compatible structured output: response_format with json_schema
        extra_kwargs: dict[str, Any] = {}
        if response_schema is not None:
            extra_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "schema": response_schema.model_json_schema(),
                    "strict": True,
                },
            }
        if max_output_tokens is not None:
            extra_kwargs["max_tokens"] = max_output_tokens

        try:
            completion = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                **extra_kwargs,
            )
        except Exception as exc:
            raise _classify_openrouter_error(exc) from exc

        if not completion.choices:
            raise LLMEmptyResponseError("openrouter returned no choices")

        choice = completion.choices[0]
        text = choice.message.content or ""
        if not text:
            raise LLMEmptyResponseError(f"empty content (finish_reason={choice.finish_reason})")

        usage = completion.usage
        return LLMResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=model,
            provider=self.name,
        )

    async def generate_stream(
        self,
        prompt: str,
        *,
        model: str,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        messages: list[ChatCompletionUserMessageParam] = [
            {"role": "user", "content": prompt},
        ]
        extra_kwargs: dict[str, Any] = {"stream_options": {"include_usage": True}}
        if max_output_tokens is not None:
            extra_kwargs["max_tokens"] = max_output_tokens

        try:
            # stream=True ⇒ AsyncStream; cast helps mypy narrow the union return.
            stream: AsyncStream[ChatCompletionChunk] = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **extra_kwargs,
            )
        except Exception as exc:
            raise _classify_openrouter_error(exc) from exc

        input_tokens = 0
        output_tokens = 0

        async for chunk in stream:
            # Usage chunks arrive separately, with no choices populated
            if chunk.usage is not None:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
                continue
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield StreamChunk(text=delta.content)

        yield StreamChunk(
            text="",
            is_final=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
