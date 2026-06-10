"""Application configuration.

All config comes from environment variables (12-factor). For local
development, values can be set in `.env` (gitignored).

Hierarchy: env var > .env > default. Pydantic-settings handles precedence.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from pydantic import Field, SecretStr, field_validator
from pydantic_core.core_schema import ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/api/config.py -> src/api -> src -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    """Top-level application configuration."""

    # ---- Application ----
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_format_json: bool = Field(default=False, description="JSON logs vs pretty.")

    # ---- Database ----
    database_url: SecretStr = Field(
        default=SecretStr("postgresql+asyncpg://app:app@localhost:5432/app"),
        description="SQLAlchemy async DSN for Postgres.",
    )
    db_echo: bool = Field(default=False)
    db_pool_size: int = Field(default=10, ge=1, le=50)
    db_max_overflow: int = Field(default=5, ge=0, le=20)

    # ---- Redis (LLM response cache) ----
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis DSN for the LLM prompt cache.",
    )
    llm_cache_ttl_seconds: float = Field(default=300.0, gt=0.0)

    # ---- External services ----
    github_api_timeout_s: float = Field(default=15.0, gt=0.0, le=60.0)
    github_max_concurrency: int = Field(default=5, ge=1, le=50)

    # ---- LLM provider secrets ----
    anthropic_api_key: SecretStr | None = Field(default=None)
    anthropic_default_model: str = Field(default="claude-haiku-4-5-20251001")
    anthropic_enable_prompt_cache: bool = Field(
        default=False,
        description="Cache system prompt + tool defs on each agent_round call.",
    )
    openai_api_key: SecretStr | None = Field(default=None)

    gemini_api_key: SecretStr | None = Field(default=None)
    gemini_default_model: str = Field(default="gemini-2.5-flash")

    openrouter_api_key: SecretStr | None = Field(default=None)
    openrouter_default_model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct:free",
        description="Default OpenRouter model. ':free' suffix uses the free tier.",
    )

    local_ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible endpoint exposed by Ollama daemon.",
    )
    local_default_model: str = Field(
        default="llama3.1:8b",
        description="Default model name for the local Ollama provider.",
    )

    llm_provider: str = Field(
        default="gemini",
        description="Which provider to use: 'gemini', 'anthropic', 'openrouter', 'local', or 'mock'.",
    )

    # ---- Agent cost guardrails ----
    # Tight defaults to keep an accidental loop on a paid provider (Anthropic,
    # OpenAI) from racking up real money. Loosen per-call via the HTTP payload
    # if you actually need a longer run.
    agent_max_cost_usd: float = Field(
        default=0.02,
        gt=0.0,
        le=5.0,
        description="Hard ceiling on USD spent per agent run.",
    )
    agent_max_iterations: int = Field(
        default=4,
        ge=1,
        le=15,
        description="Maximum tool-use iterations before the agent bails out.",
    )
    agent_max_output_tokens: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="max_tokens on each LLM round inside the agent loop.",
    )

    # ---- Behavior ----
    rate_limit_max_requests: int = Field(default=10, ge=1)
    rate_limit_window_seconds: float = Field(default=60.0, gt=0.0)
    score_cache_ttl_seconds: float = Field(default=60.0, gt=0.0)

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",  # ← reject unknown env vars
    )

    @field_validator("debug")
    @classmethod
    def debug_never_in_production(cls, v: bool, info: ValidationInfo) -> bool:
        """Refuse to start with debug=True in production."""
        env = info.data.get("environment")
        if v and env == Environment.PRODUCTION:
            raise ValueError("debug=True is forbidden in production")
        return v


@lru_cache
def get_settings() -> Settings:
    """Cached settings (one instance per process)."""
    return Settings()


def get_settings_dep() -> Settings:
    """FastAPI dependency wrapper around get_settings."""
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
