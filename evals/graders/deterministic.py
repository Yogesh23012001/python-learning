"""Deterministic graders — apply rules in the dataset schema, no LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeterministicScore:
    """Result of deterministic grading for one example."""

    # Per-check pass/fail
    tool_selection_ok: bool  # expected tools called?
    must_contain_ok: bool  # all required substrings present?
    must_not_contain_ok: bool  # all red flags absent?
    outcome_ok: bool  # actual outcome matches expected?

    # Aggregate
    score: float  # 0.0 - 1.0 based on weighted check pass rate
    notes: list[str]  # human-readable explanation per failure


def grade_deterministic(
    *,
    example: dict[str, Any],
    actual_tools: list[str],
    actual_text: str,
    actual_outcome: str,
) -> DeterministicScore:
    """Apply schema rules to one (example, run) pair.

    example: the dataset entry being graded
    actual_tools: tool names called by the agent, in order
    actual_text: the final text response
    actual_outcome: 'completed', 'max_iterations', 'cost_cap', 'tool_error', etc.
    """
    notes: list[str] = []

    # ---- Check 1: expected_tools ----
    expected_tools = example.get("expected_tools")
    if expected_tools is None:
        # null means "don't grade tool selection"
        tool_selection_ok = True
    else:
        # Multiset comparison: expected and actual must match (order-insensitive)
        # but allow extra tool calls (agent may chain more than necessary).
        expected_set = set(expected_tools)
        actual_set = set(actual_tools)
        missing = expected_set - actual_set
        if missing:
            tool_selection_ok = False
            notes.append(f"missing expected tools: {sorted(missing)}")
        else:
            tool_selection_ok = True

    # ---- Check 2: must_contain ----
    must_contain = example.get("must_contain", [])
    missing_strings = [s for s in must_contain if s.lower() not in actual_text.lower()]
    if missing_strings:
        must_contain_ok = False
        notes.append(f"response missing required strings: {missing_strings}")
    else:
        must_contain_ok = True

    # ---- Check 3: must_not_contain ----
    must_not_contain = example.get("must_not_contain", [])
    forbidden_found = [s for s in must_not_contain if s.lower() in actual_text.lower()]
    if forbidden_found:
        must_not_contain_ok = False
        notes.append(f"response contains forbidden strings: {forbidden_found}")
    else:
        must_not_contain_ok = True

    # ---- Check 4: expected_outcome ----
    expected_outcome = example.get("expected_outcome", "completed")
    outcome_ok = actual_outcome == expected_outcome
    if not outcome_ok:
        notes.append(f"outcome mismatch: expected {expected_outcome!r}, got {actual_outcome!r}")

    # ---- Aggregate score ----
    # Weighted: tool selection 0.4, content checks 0.3 each, outcome 0.0 (already captured)
    # Adjust weights based on your priorities.
    weights = {
        "tool_selection": 0.40,
        "must_contain": 0.30,
        "must_not_contain": 0.30,
    }
    score = (
        weights["tool_selection"] * (1.0 if tool_selection_ok else 0.0)
        + weights["must_contain"] * (1.0 if must_contain_ok else 0.0)
        + weights["must_not_contain"] * (1.0 if must_not_contain_ok else 0.0)
    )

    return DeterministicScore(
        tool_selection_ok=tool_selection_ok,
        must_contain_ok=must_contain_ok,
        must_not_contain_ok=must_not_contain_ok,
        outcome_ok=outcome_ok,
        score=score,
        notes=notes,
    )
