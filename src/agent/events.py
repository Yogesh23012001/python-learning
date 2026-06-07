"""Event types emitted by the agent loop.

The agent loop is a generator that yields these events as they happen.
Consumers can render them however they want — print to CLI, push as SSE
to a browser, write to a database, fan out to multiple subscribers.

Design principle: events are pure data, no behavior. They're a wire
format between the agent loop and its consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ============================================================
# Event types
# ============================================================


@dataclass(frozen=True)
class AgentStarted:
    """Emitted once at the start of every agent run."""

    type: Literal["agent_started"] = "agent_started"
    prompt_length: int = 0
    max_iterations: int = 0
    model: str = ""
    request_id: str = ""


@dataclass(frozen=True)
class IterationStarted:
    """Emitted at the start of each loop iteration."""

    iteration: int
    type: Literal["iteration_started"] = "iteration_started"
    request_id: str = ""


@dataclass(frozen=True)
class ToolRequested:
    """Emitted when the LLM requests a tool call (before execution)."""

    name: str
    args: dict[str, Any]
    iteration: int
    type: Literal["tool_requested"] = "tool_requested"
    request_id: str = ""


@dataclass(frozen=True)
class ToolCompleted:
    """Emitted after a tool runs successfully."""

    name: str
    iteration: int
    result_keys: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    type: Literal["tool_completed"] = "tool_completed"
    request_id: str = ""


@dataclass(frozen=True)
class ToolFailed:
    """Emitted when a tool raises or times out."""

    name: str
    iteration: int
    error: str = ""
    error_type: Literal["timeout", "refused", "exception"] = "exception"
    type: Literal["tool_failed"] = "tool_failed"
    request_id: str = ""


@dataclass(frozen=True)
class AgentCompleted:
    """Emitted as the final event on the normal exit path."""

    text: str = ""
    iterations: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    type: Literal["agent_completed"] = "agent_completed"
    request_id: str = ""


@dataclass(frozen=True)
class AgentStopped:
    """Emitted as the final event when a safety cap fired."""

    reason: Literal["max_iterations", "cost_cap"]
    iterations: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    type: Literal["agent_stopped"] = "agent_stopped"
    request_id: str = ""


# Union type for all events
AgentEvent = (
    AgentStarted
    | IterationStarted
    | ToolRequested
    | ToolCompleted
    | ToolFailed
    | AgentCompleted
    | AgentStopped
)
