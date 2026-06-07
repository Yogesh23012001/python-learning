"""SQLAlchemy ORM models."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from api.db import Base


class StoredScore(Base):
    """Persisted developer scores for GitHub users."""

    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    login: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    total_stars: Mapped[int] = mapped_column(BigInteger)
    total_forks: Mapped[int] = mapped_column(BigInteger)
    public_repos: Mapped[int] = mapped_column(Integer)
    followers: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column()
    score_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="1",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    summary: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    summary_generated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class AgentRun(Base):
    """Append-only audit row per agent run."""

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    prompt: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(128))
    provider: Mapped[str] = mapped_column(String(64))
    iterations: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(10, 6), default=Decimal("0"))
    outcome: Mapped[str] = mapped_column(String(32))
    text_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
