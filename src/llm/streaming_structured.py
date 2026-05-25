"""Streaming token-by-token from Gemini."""

from __future__ import annotations

from enum import StrEnum

from api.config import get_settings
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ReviewAnalysis(BaseModel):
    sentiment: Sentiment
    score: int = Field(ge=1, le=5, description="Overall rating 1-5")
    key_complaints: list[str] = Field(description="Specific complaints, empty if none")
    key_praises: list[str] = Field(description="Specific things praised, empty if none")
    summary: str = Field(description="One-sentence summary")


settings = get_settings()
client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

response_stream = client.models.generate_content_stream(
    model=settings.gemini_default_model,
    contents="Generate a product review for analysis",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=ReviewAnalysis,
    ),
)

print("raw streaming chunks:")
chunks = []
for chunk in response_stream:
    if chunk.text:
        chunks.append(chunk.text)
        print(chunk.text, end="", flush=True)
print()

# After streaming completes, parse the full text into the Pydantic model
full_json = "".join(chunks)
analysis = ReviewAnalysis.model_validate_json(full_json)
print("\nparsed:")
print(analysis.model_dump_json(indent=2))
