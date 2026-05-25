"""SQLAlchemy ORM models."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import BigInteger, DateTime, Integer, String
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
