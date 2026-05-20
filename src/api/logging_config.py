"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(
    *,
    level: int = logging.INFO,
    json_logs: bool = True,
) -> None:
    """Configure both stdlib logging and structlog to share output.

    Args:
        level: stdlib log level (DEBUG, INFO, WARNING, ...).
        json_logs: True for JSON output (production); False for pretty (dev).
    """
    # Configure structlog's processors
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,  # ← THE pattern (see below)
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        timestamper,
    ]

    renderer: Any
    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Send stdlib logs through structlog too — unifies output
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
