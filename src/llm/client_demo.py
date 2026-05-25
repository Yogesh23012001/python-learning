"""Demo the production LLMClient."""

from __future__ import annotations

import asyncio
from enum import StrEnum

from api.config import get_settings
from api.logging_config import configure_logging
from pydantic import BaseModel, Field

from llm.client import LLMClient
from llm.errors import LLMError


class Intent(StrEnum):
    QUESTION = "question"
    COMMAND = "command"
    GREETING = "greeting"
    OTHER = "other"


class IntentClassification(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


async def main() -> None:
    configure_logging(json_logs=False, level=20)
    settings = get_settings()
    client = LLMClient(
        api_key=settings.gemini_api_key.get_secret_value(),
        default_model=settings.gemini_default_model,
    )

    # 1. Simple text call
    print("=== Simple call ===")
    resp = await client.generate("In 5 words, what is AI?")
    print(f"text: {resp.text}")
    print(f"cost: ${resp.cost_usd:.6f}  ({resp.input_tokens} in / {resp.output_tokens} out)")
    print()

    # 2. Structured output
    print("=== Structured ===")
    inputs = [
        "What time is it in Tokyo?",
        "Turn off the kitchen lights",
        "Hey there!",
        "lkasjdflkj",
    ]
    total_cost = 0.0
    for inp in inputs:
        try:
            resp = await client.generate(
                f"Classify this user message: {inp!r}",
                response_schema=IntentClassification,
            )
            parsed = IntentClassification.model_validate_json(resp.text)
            total_cost += resp.cost_usd
            print(f"  {inp!r:<35} → {parsed.intent.value} (confidence={parsed.confidence:.2f})")
        except LLMError as exc:
            print(f"  {inp!r:<35} → ERROR: {type(exc).__name__}: {exc}")
    print(f"\ntotal cost: ${total_cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
