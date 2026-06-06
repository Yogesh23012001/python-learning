"""Build a configured LLMRouter from settings."""

from __future__ import annotations

from api.config import Settings

from llm.providers.gemini import GeminiProvider
from llm.providers.local_ollama import LocalOllamaProvider
from llm.providers.mock import MockProvider
from llm.providers.openrouter import OpenRouterProvider
from llm.router import LLMRouter


def make_router(settings: Settings, *, use_mock: bool = False) -> LLMRouter:
    """Construct an LLMRouter wired to the configured provider."""
    if use_mock or settings.llm_provider == "mock":
        return LLMRouter(provider=MockProvider(), default_model="mock-model-1")

    if settings.llm_provider == "openrouter":
        if settings.openrouter_api_key is None:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        return LLMRouter(
            provider=OpenRouterProvider(
                api_key=settings.openrouter_api_key.get_secret_value(),
            ),
            default_model=settings.openrouter_default_model,
        )

    if settings.llm_provider == "gemini":
        if settings.gemini_api_key is None:
            raise RuntimeError("GEMINI_API_KEY not set")
        return LLMRouter(
            provider=GeminiProvider(
                api_key=settings.gemini_api_key.get_secret_value(),
            ),
            default_model=settings.gemini_default_model,
        )

    if settings.llm_provider == "local":
        return LLMRouter(
            provider=LocalOllamaProvider(base_url=settings.local_ollama_base_url),
            default_model=settings.local_default_model,
        )

    raise RuntimeError(f"unknown llm_provider: {settings.llm_provider!r}")
