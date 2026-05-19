"""Database engine, session, and dependency for FastAPI."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from api.config import get_settings


class Base(DeclarativeBase):
    """Base class for all ORM models."""


def make_engine() -> AsyncEngine:
    """Create the async SQLAlchemy engine (one per process)."""
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,  # check connection liveness on checkout
    )


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build a session factory bound to the engine."""
    return async_sessionmaker(
        bind=engine,
        expire_on_commit=False,  # objects stay usable after commit
        autoflush=False,
    )


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields a session per request, cleans up on exit."""
    session_factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


DbDep = Annotated[AsyncSession, Depends(get_db)]
