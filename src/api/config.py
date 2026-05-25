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

    # ---- External services ----
    github_api_timeout_s: float = Field(default=15.0, gt=0.0, le=60.0)
    github_max_concurrency: int = Field(default=5, ge=1, le=50)

    # ---- Future: LLM provider secrets ----
    anthropic_api_key: SecretStr | None = Field(default=None)
    openai_api_key: SecretStr | None = Field(default=None)

    gemini_api_key: SecretStr | None = Field(default=None)
    gemini_default_model: str = Field(default="gemini-2.5-flash")

    openrouter_api_key: SecretStr | None = Field(default=None)
    openrouter_default_model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct:free",
        description="Default OpenRouter model. ':free' suffix uses the free tier.",
    )

    llm_provider: str = Field(
        default="gemini",
        description="Which provider to use: 'gemini', 'openrouter', or 'mock'.",
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
