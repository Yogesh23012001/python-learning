"""Hour 2 — token counting basics."""

from __future__ import annotations

from api.config import get_settings
from google import genai


def main() -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    samples = [
        "Hello",
        "What is the capital of France?",
        "Yogesh works at NPCI on payment infrastructure",
        "    ",  # whitespace
        "🚀🌟⭐",  # emoji
        "https://api.anthropic.com/v1/messages",  # URL
        "def hello():\n    print('world')\n",  # code
    ]

    for s in samples:
        result = client.models.count_tokens(
            model=settings.gemini_default_model,
            contents=s,
        )
        print(
            f"{result.total_tokens:>4} tokens | {len(s):>4} chars | ratio {len(s) / max(result.total_tokens, 1):>5.2f} c/t | {s!r}"
        )


if __name__ == "__main__":
    main()
