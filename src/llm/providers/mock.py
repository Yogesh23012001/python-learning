"""Mock provider for tests and offline development."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from llm.errors import LLMRateLimitError
from llm.interface import LLMResponse, StreamChunk


class MockProvider:
    """Provider that returns deterministic-ish fake responses.

    Use cases:
      - Unit tests that need predictable LLM behavior
      - Local development without burning real quota
      - Integration tests that should never hit the network
    """

    name = "mock"

    def __init__(
        self,
        *,
        fixed_response: str | None = None,
        simulate_latency_ms: float = 50.0,
        failure_rate: float = 0.0,
        failure_type: type[Exception] = LLMRateLimitError,
    ) -> None:
        self._fixed_response = fixed_response
        self._latency = simulate_latency_ms / 1000.0
        self._failure_rate = failure_rate
        self._failure_type = failure_type
        self._rng = random.Random(42)

    def _maybe_fail(self) -> None:
        if self._failure_rate > 0 and self._rng.random() < self._failure_rate:
            raise self._failure_type(f"mock: simulated {self._failure_type.__name__}")

    def _response_for(self, prompt: str) -> str:
        if self._fixed_response is not None:
            return self._fixed_response
        # Deterministic-ish based on prompt
        return f"[mock] you said: {prompt[:50]}..."

    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        response_schema: type[BaseModel] | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        await asyncio.sleep(self._latency)
        self._maybe_fail()
        text = self._response_for(prompt)
        # If a schema is requested, the mock has to produce valid JSON for it
        if response_schema is not None:
            # Build a minimal instance — every field gets a default-ish value
            mock_instance = _mock_instance_for_schema(response_schema)
            text = mock_instance.model_dump_json()
        return LLMResponse(
            text=text,
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
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
        self._maybe_fail()
        text = self._response_for(prompt)
        # Stream word by word with small delays for realism
        words = text.split(" ")
        for i, word in enumerate(words):
            await asyncio.sleep(self._latency / 10)
            chunk_text = word if i == 0 else f" {word}"
            yield StreamChunk(text=chunk_text)
        yield StreamChunk(
            text="",
            is_final=True,
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
        )


def _mock_instance_for_schema(schema: type[BaseModel]) -> BaseModel:
    """Build a Pydantic model instance with default-ish values for each field."""
    values: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        if field.default is not None and not (
            hasattr(field.default, "__class__")
            and field.default.__class__.__name__ == "PydanticUndefinedType"
        ):
            continue  # respect existing defaults
        annotation = field.annotation
        if annotation is None:
            values[name] = None
            continue
        if annotation is str:
            values[name] = "mock-value"
        elif annotation is int:
            values[name] = 0
        elif annotation is float:
            values[name] = 0.5
        elif annotation is bool:
            values[name] = False
        elif hasattr(annotation, "__origin__") and annotation.__origin__ is list:
            values[name] = []
        else:
            # Best-effort: enums get their first value. We can't statically
            # prove `annotation` is iterable (it's just a type[Any]); enums
            # happen to be at runtime. TypeError is caught for non-iterables.
            try:
                values[name] = next(iter(annotation))  # type: ignore[call-overload]
            except (TypeError, StopIteration):
                values[name] = None
    return schema(**values)
