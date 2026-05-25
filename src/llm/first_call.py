"""Hour 1 — first real Gemini call."""

from __future__ import annotations

from api.config import get_settings
from google import genai


def main() -> None:
    settings = get_settings()
    if settings.gemini_api_key is None:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    # The SDK reads GEMINI_API_KEY from environment, OR you pass api_key explicitly
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    response = client.models.generate_content(
        model=settings.gemini_default_model,
        contents="In one short sentence, what is the capital of France?",
    )

    print("response:")
    print(response.text)
    print()
    print("usage:")
    print(f"  input tokens:  {response.usage_metadata.prompt_token_count}")
    print(f"  output tokens: {response.usage_metadata.candidates_token_count}")
    print(f"  total tokens:  {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    main()
