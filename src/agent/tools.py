"""Tool definitions, schemas, and dispatch for the agent system.

A "tool" is a Python function the LLM can request the system to call. Each
tool has three parts:
  1. The implementation (a regular Python function)
  2. A JSON schema describing it to the LLM
  3. Registration in HANDLERS for dispatch by name

The pattern is identical to the worker handler registry in
idempotent-task-queue — same engineering shape, different domain.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ============================================================
# Types
# ============================================================


ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


# ============================================================
# Tool implementations
# ============================================================


async def get_current_time(args: dict[str, Any]) -> dict[str, Any]:
    """Return the current UTC time as ISO 8601."""
    now = datetime.now(UTC).isoformat()
    return {"utc_now": now}


# ============================================================
# Tool schemas (described to the LLM)
# ============================================================


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_current_time",
        "description": (
            "Return the current UTC time as an ISO 8601 string. "
            "Use when the user asks about the current time, today's date, "
            "or any question that requires knowing 'now'."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ============================================================
# Registry + dispatch
# ============================================================


HANDLERS: dict[str, ToolHandler] = {
    "get_current_time": get_current_time,
}


class ToolNotFoundError(Exception):
    """Raised when the LLM requests a tool we don't have."""


async def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call by name."""
    handler = HANDLERS.get(name)
    if handler is None:
        raise ToolNotFoundError(f"no tool registered with name={name!r}")

    logger.info("tool_call_started", tool=name, args=args)
    try:
        result = await handler(args)
    except Exception as exc:
        logger.warning("tool_call_failed", tool=name, error=str(exc)[:200])
        raise
    logger.info(
        "tool_call_completed",
        tool=name,
        result_keys=list(result.keys()),
    )
    return result
