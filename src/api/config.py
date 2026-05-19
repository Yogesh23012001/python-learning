"""Application configuration loaded from environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor .env to the project root, not the cwd that launched Python.
# src/api/config.py -> src/api -> src -> myproject
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application configuration."""

    database_url: str = Field(
        default="postgresql+asyncpg://app:app@localhost:5432/app",
        description="SQLAlchemy async DSN for Postgres.",
    )
    db_echo: bool = Field(
        default=False,
        description="If True, log every SQL statement (debug only).",
    )
    db_pool_size: int = Field(default=10, ge=1)
    db_max_overflow: int = Field(default=5, ge=0)

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings — loaded once per process."""
    return Settings()
