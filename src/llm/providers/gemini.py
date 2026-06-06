"""Gemini provider adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

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
from llm.interface import (
    AgentMessage,
    AgentRoundResponse,
    LLMResponse,
    StreamChunk,
    ToolCall,
)


def _classify_gemini_error(exc: Exception) -> LLMError:
    """Convert Gemini-specific exceptions into shared typed errors.

    This is the ONLY place in the codebase that knows about Gemini's error shapes.
    """
    if isinstance(exc, genai_errors.ClientError):
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        message = str(exc).lower()
        if status == 429:
            if "quota" in message:
                return LLMQuotaExceededError(_short(exc))
            return LLMRateLimitError(_short(exc))
        if status == 400 and ("context" in message or "token" in message):
            return LLMContextTooLongError(_short(exc))
        if status == 503:
            return LLMOverloadedError(_short(exc))

    if isinstance(exc, genai_errors.ServerError):
        return LLMOverloadedError(_short(exc))

    return LLMRetryableError(_short(exc))


def _short(exc: Exception) -> str:
    """Trim multi-line, multi-KB error messages to something log-friendly."""
    return str(exc).split("\n")[0][:200]


class GeminiProvider:
    """Adapter for Google's Gemini API."""

    name = "gemini"

    def __init__(self, *, api_key: str) -> None:
        self._client = genai.Client(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        response_schema: type[BaseModel] | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        config_dict: dict[str, Any] = {}
        if max_output_tokens is not None:
            config_dict["max_output_tokens"] = max_output_tokens
        if response_schema is not None:
            config_dict["response_mime_type"] = "application/json"
            config_dict["response_schema"] = response_schema

        try:
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_dict) if config_dict else None,
            )
        except Exception as exc:
            raise _classify_gemini_error(exc) from exc

        if not response.candidates:
            raise LLMEmptyResponseError("model returned no candidates")
        candidate = response.candidates[0]
        if candidate.finish_reason == types.FinishReason.SAFETY:
            raise LLMContentBlockedError(f"safety filter triggered: {candidate.safety_ratings}")
        if not response.text:
            raise LLMEmptyResponseError(f"empty text (finish_reason={candidate.finish_reason})")

        usage = response.usage_metadata
        return LLMResponse(
            text=response.text,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
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
        config_dict: dict[str, Any] = {}
        if max_output_tokens is not None:
            config_dict["max_output_tokens"] = max_output_tokens

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_dict) if config_dict else None,
            )
        except Exception as exc:
            raise _classify_gemini_error(exc) from exc

        input_tokens = 0
        output_tokens = 0
        async for chunk in stream:
            if chunk.text:
                yield StreamChunk(text=chunk.text)
            if chunk.usage_metadata:
                input_tokens = chunk.usage_metadata.prompt_token_count or 0
                output_tokens = chunk.usage_metadata.candidates_token_count or 0

        # Final marker carrying usage
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
        """Translate neutral messages → Gemini Contents, run one round."""
        gemini_contents = [_to_gemini_content(m) for m in messages]
        gemini_tools: list[types.Tool | Any] = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(**schema) for schema in tool_schemas
                ]
            )
        ]
        config_dict: dict[str, Any] = {"tools": gemini_tools}
        if max_output_tokens is not None:
            config_dict["max_output_tokens"] = max_output_tokens

        try:
            response = await self._client.aio.models.generate_content(
                model=model,
                contents=gemini_contents,
                config=types.GenerateContentConfig(**config_dict),
            )
        except Exception as exc:
            raise _classify_gemini_error(exc) from exc

        if not response.candidates:
            raise LLMEmptyResponseError("gemini returned no candidates")
        candidate = response.candidates[0]
        if candidate.content is None:
            raise LLMEmptyResponseError("gemini candidate had no content")
        parts = candidate.content.parts or []

        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        for idx, part in enumerate(parts):
            if part.text:
                text_chunks.append(part.text)
            if part.function_call is not None and part.function_call.name is not None:
                tool_calls.append(
                    ToolCall(
                        id=f"call_{part.function_call.name}_{idx}",
                        name=part.function_call.name,
                        args=dict(part.function_call.args) if part.function_call.args else {},
                    )
                )

        usage = response.usage_metadata
        return AgentRoundResponse(
            text="".join(text_chunks),
            tool_calls=tool_calls,
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        )


def _to_gemini_content(m: AgentMessage) -> types.Content:
    """Translate one neutral AgentMessage into a Gemini Content object."""
    if m.role == "user":
        return types.Content(role="user", parts=[types.Part.from_text(text=m.content)])

    if m.role == "assistant":
        parts: list[types.Part] = []
        if m.content:
            parts.append(types.Part(text=m.content))
        for tc in m.tool_calls:
            parts.append(types.Part(function_call=types.FunctionCall(name=tc.name, args=tc.args)))
        return types.Content(role="model", parts=parts)

    if m.role == "tool":
        # Gemini uses the function NAME, not an id, to match a response back.
        name = m.tool_name or ""
        # m.content is a JSON-string of the tool result; Gemini wants a dict.
        import json as _json

        try:
            response_dict = _json.loads(m.content) if m.content else {}
        except _json.JSONDecodeError:
            response_dict = {"raw": m.content}
        return types.Content(
            role="user",
            parts=[types.Part.from_function_response(name=name, response=response_dict)],
        )

    raise ValueError(f"unknown AgentMessage role: {m.role!r}")
