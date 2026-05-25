"""Streaming token-by-token from Gemini."""

from __future__ import annotations

from api.config import get_settings
from google import genai


def main() -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    prompt = "Write a 4-sentence story about a backend engineer learning AI infrastructure."

    print(f"prompt: {prompt}\n")
    print("response (streaming):")

    response_stream = client.models.generate_content_stream(
        model=settings.gemini_default_model,
        contents=prompt,
    )

    for chunk in response_stream:
        if chunk.text:
            print(chunk.text, end="", flush=True)

    print()  # final newline


if __name__ == "__main__":
    main()
