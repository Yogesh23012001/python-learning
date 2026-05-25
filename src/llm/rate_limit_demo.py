import asyncio

from api.config import get_settings
from api.logging_config import configure_logging

from llm.client import LLMClient


async def main() -> None:
    configure_logging(json_logs=False, level=20)
    settings = get_settings()
    client = LLMClient(
        api_key=settings.gemini_api_key.get_secret_value(),
        default_model=settings.gemini_default_model,
    )

    # 20 concurrent calls — should exceed 15/min free tier limit
    async def one_call(i: int) -> tuple[int, str]:
        try:
            resp = await client.generate(f"Say the number {i} in one word")
            return (i, f"OK: {resp.text.strip()[:30]}")
        except Exception as exc:
            return (i, f"ERR: {type(exc).__name__}")

    results = await asyncio.gather(*[one_call(i) for i in range(20)])
    for i, msg in sorted(results):
        print(f"call {i:2}: {msg}")


if __name__ == "__main__":
    asyncio.run(main())
