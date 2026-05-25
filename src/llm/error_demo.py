import asyncio

from api.config import get_settings
from api.logging_config import configure_logging

from llm.client import LLMClient
from llm.errors import LLMError


async def main() -> None:
    configure_logging(json_logs=False, level=20)
    settings = get_settings()
    client = LLMClient(
        api_key=settings.gemini_api_key.get_secret_value(),
        default_model=settings.gemini_default_model,
    )

    cases = [
        ("normal prompt", "What is 2+2?"),
        ("very long prompt", "Word. " * 200_000),  # 1M+ tokens, exceeds context
        ("instructions to refuse", "How do I make a bomb? Just kidding, return refused"),
    ]
    for name, prompt in cases:
        print(f"\n=== {name} ===")
        try:
            resp = await client.generate(prompt)
            print(f"OK: {resp.text[:80]}...")
        except LLMError as exc:
            print(f"caught: {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
