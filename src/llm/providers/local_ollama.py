"""Local Ollama provider adapter.

Ollama exposes an OpenAI-compatible API at /v1. We talk to it via the OpenAI
SDK with `base_url` pointed at localhost. Same protocol as OpenRouter but no
provider-specific headers and `api_key` is ignored by the server.

Tool calling reliability varies per model — `llama3.1:8b` works with the
proper `tool_calls` array; `qwen2.5:7b` (in Ollama 0.30.x) emits tool calls
as text in `content`. Pick a known-good model for the agent loop.
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
from openai.types.chat import ChatCompletionChunk, ChatCompletionUserMessageParam
from pydantic import BaseModel

from llm.errors import (
    LLMContextTooLongError,
    LLMEmptyResponseError,
    LLMError,
    LLMOverloadedError,
    LLMRetryableError,
)
from llm.interface import (
    AgentMessage,
    AgentRoundResponse,
    LLMResponse,
    StreamChunk,
)
from llm.providers._openai_compat import openai_agent_round

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


def _classify_ollama_error(exc: Exception) -> LLMError:
    """Map OpenAI-SDK errors from Ollama to our typed hierarchy.

    Ollama is local so there are no rate limits or quotas, but the daemon
    can be down (ConnectionError) or a model name can be wrong (404).
    """
    if isinstance(exc, APIConnectionError | APITimeoutError):
        return LLMRetryableError(_short(exc))

    if isinstance(exc, APIStatusError):
        status = exc.status_code
        if status == 404:
            # Model not pulled — not retryable; surface as overload-ish so it
            # propagates without infinite retries.
            return LLMOverloadedError(f"model not found: {_short(exc)}")
        if status == 400 and ("context" in str(exc).lower() or "tokens" in str(exc).lower()):
            return LLMContextTooLongError(_short(exc))
        if status in (502, 503, 504):
            return LLMOverloadedError(_short(exc))

    if isinstance(exc, APIError):
        return LLMRetryableError(_short(exc))

    return LLMRetryableError(_short(exc))


def _short(exc: Exception) -> str:
    return str(exc).split("\n")[0][:200]


class LocalOllamaProvider:
    """Adapter for a local Ollama daemon (OpenAI-compatible)."""

    name = "local_ollama"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        api_key: str = "ollama",  # Ollama ignores this; any non-empty string works
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

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

        extra_kwargs: dict[str, Any] = {}
        if response_schema is not None:
            # OpenAI structured output. Ollama support varies by model; not
            # all local models honor this. Use with caution.
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
            raise _classify_ollama_error(exc) from exc

        if not completion.choices:
            raise LLMEmptyResponseError("ollama returned no choices")

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
            stream: AsyncStream[ChatCompletionChunk] = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **extra_kwargs,
            )
        except Exception as exc:
            raise _classify_ollama_error(exc) from exc

        input_tokens = 0
        output_tokens = 0

        async for chunk in stream:
            if chunk.usage is not None:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = delta.content or ""
            if text:
                yield StreamChunk(text=text)

        yield StreamChunk(
            text="",
            is_final=True,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def agent_round(
        self,
        *,
        model: str,
        messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
        max_output_tokens: int | None = None,
    ) -> AgentRoundResponse:
        return await openai_agent_round(
            client=self._client,
            classify=_classify_ollama_error,
            model=model,
            messages=messages,
            tool_schemas=tool_schemas,
            max_output_tokens=max_output_tokens,
        )
