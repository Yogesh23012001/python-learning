"""Curated canary prompts that exercise specific failure modes.

Each canary has:
  - name: filename-safe identifier
  - prompt: the user input
  - expected_tools: which tools SHOULD be called (in any order)
  - expected_failure_mode: None if expected to succeed, otherwise one of:
       "wrong_tool" | "iteration_cap" | "tool_error_unrecovered" |
       "wrong_reasoning" | "cost_runaway"

These are run by capture_traces.py to produce traces in traces/<name>.json
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanaryPrompt:
    name: str
    prompt: str
    expected_tools: list[str]
    expected_failure_mode: str | None = None
    max_iterations: int = 8
    max_cost_usd: float = 0.10


CANARIES: list[CanaryPrompt] = [
    # ============================================================
    # Happy paths — should always work
    # ============================================================
    CanaryPrompt(
        name="happy_current_time",
        prompt="What time is it in UTC right now?",
        expected_tools=["get_current_time"],
    ),
    CanaryPrompt(
        name="happy_github_lookup",
        prompt="What is karpathy's GitHub score?",
        expected_tools=["lookup_github_user"],
    ),
    CanaryPrompt(
        name="happy_database_query",
        prompt="Show me the developers we have records of",
        expected_tools=["query_stored_scores"],
    ),
    CanaryPrompt(
        name="happy_math",
        prompt="What's 462453.7 divided by 84577.4 to 3 decimal places?",
        expected_tools=["calculate"],
    ),
    CanaryPrompt(
        name="happy_multi_tool_comparison",
        prompt="Compare karpathy and torvalds GitHub scores and tell me the ratio",
        expected_tools=["lookup_github_user", "lookup_github_user", "calculate"],
    ),
    # ============================================================
    # Refusal — should NOT call any tool
    # ============================================================
    CanaryPrompt(
        name="refusal_no_relevant_tool",
        prompt="What is the current state of the European stock market?",
        expected_tools=[],
    ),
    # ============================================================
    # Known failures — documented expected behavior
    # ============================================================
    CanaryPrompt(
        name="failure_disambiguation_linus",
        prompt="Tell me about the GitHub user linus",
        expected_tools=["lookup_github_user"],
        # This one calls the right tool but produces the wrong person (Linus G Thiel)
        expected_failure_mode="wrong_reasoning",
    ),
    CanaryPrompt(
        name="failure_tool_error_user_not_exist",
        prompt="Add the user yannickcollet-does-not-exist to our database",
        expected_tools=["score_and_save_user"],
        expected_failure_mode="tool_error_unrecovered",
    ),
    # ============================================================
    # Safety boundary tests
    # ============================================================
    CanaryPrompt(
        name="safety_iteration_cap",
        prompt=(
            "Call get_current_time, then lookup_github_user for torvalds, "
            "then lookup_github_user for antirez, then calculate their ratio, "
            "then call get_current_time again, then lookup torvalds again, "
            "doing each step separately"
        ),
        expected_tools=[],
        expected_failure_mode="iteration_cap",
        max_iterations=2,
    ),
    CanaryPrompt(
        name="safety_cost_cap",
        prompt="Compare karpathy and torvalds scores in extensive detail",
        expected_tools=[],
        expected_failure_mode="cost_runaway",
        max_cost_usd=0.00001,
    ),
]
