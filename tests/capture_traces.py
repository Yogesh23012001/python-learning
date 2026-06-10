"""Run canary prompts and capture their event traces to JSON files.

Usage:
    uv run python -m tests.capture_traces

Outputs:
    traces/<canary_name>.json — one file per canary

Each trace file contains:
    {
      "canary": { ... canary metadata ... },
      "events": [ ... list of events as dicts ... ],
      "verdict": "ok" | "regression" | "expected_failure"
    }
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

from agent.events import (
    AgentCompleted,
    AgentStopped,
    ToolRequested,
)
from agent.loop import run_agent_stream
from agent.tools import ToolContext
from api.config import get_settings
from api.db import make_engine, make_session_factory
from github_fetcher.client import GitHubClient
from llm.factory import make_router
from llm.router import LLMRouter

from tests.canary_prompts import CANARIES, CanaryPrompt

TRACES_DIR = Path(__file__).parent.parent / "traces"
TRACES_DIR.mkdir(exist_ok=True)


async def run_canary(canary: CanaryPrompt, context: ToolContext, llm: LLMRouter) -> dict:
    """Run one canary, return a trace dict ready for JSON serialization."""
    events: list[dict] = []
    actual_tools: list[str] = []
    terminal_event = None

    try:
        async for event in run_agent_stream(
            canary.prompt,
            llm=llm,
            tool_context=context,
            max_iterations=canary.max_iterations,
            max_cost_usd=canary.max_cost_usd,
        ):
            events.append(
                {"type": event.type, **{k: v for k, v in asdict(event).items() if k != "type"}}
            )
            if isinstance(event, ToolRequested):
                actual_tools.append(event.name)
            if isinstance(event, AgentCompleted | AgentStopped):
                terminal_event = event
    except Exception as exc:
        return {
            "canary": asdict(canary),
            "events": events,
            "actual_tools": actual_tools,
            "verdict": "crash",
            "error": str(exc)[:300],
        }

    # Determine verdict
    verdict = _judge(canary, actual_tools, terminal_event)

    return {
        "canary": asdict(canary),
        "events": events,
        "actual_tools": actual_tools,
        "terminal": _summarize_terminal(terminal_event),
        "verdict": verdict,
    }


def _judge(
    canary: CanaryPrompt,
    actual_tools: list[str],
    terminal: AgentCompleted | AgentStopped | None,
) -> str:
    """Compare actual behavior to expected. Return verdict string."""
    # Cost runaway / iteration cap expected
    if canary.expected_failure_mode == "iteration_cap":
        if isinstance(terminal, AgentStopped) and terminal.reason == "max_iterations":
            return "ok_expected_failure"
        return "regression_safety_didnt_fire"

    if canary.expected_failure_mode == "cost_runaway":
        if isinstance(terminal, AgentStopped) and terminal.reason == "cost_cap":
            return "ok_expected_failure"
        return "regression_safety_didnt_fire"

    if canary.expected_failure_mode == "wrong_reasoning":
        # We can't auto-detect wrong reasoning — note that human review needed
        return "ok_known_failure_human_review_required"

    if canary.expected_failure_mode == "tool_error_unrecovered":
        if isinstance(terminal, AgentCompleted) and any(
            t in canary.expected_tools for t in actual_tools
        ):
            return "ok_known_failure"
        return "unexpected"

    # Happy path — expect right tools called, agent completed
    if not isinstance(terminal, AgentCompleted):
        return "regression_didnt_complete"

    # Check expected tools (multiset compare — order-insensitive)
    expected_sorted = sorted(canary.expected_tools)
    actual_sorted = sorted(actual_tools)
    if expected_sorted != actual_sorted:
        # Allow extra tool calls (agent may chain more than necessary),
        # but flag if expected tools are missing
        missing = set(expected_sorted) - set(actual_sorted)
        if missing:
            return f"regression_missing_tools:{','.join(sorted(missing))}"

    return "ok"


def _summarize_terminal(terminal) -> dict | None:
    if terminal is None:
        return None
    return {
        "type": terminal.type,
        "iterations": terminal.iterations,
        "total_cost_usd": round(terminal.total_cost_usd, 6),
        **({"reason": terminal.reason} if isinstance(terminal, AgentStopped) else {}),
        **({"text_preview": terminal.text[:200]} if isinstance(terminal, AgentCompleted) else {}),
    }


async def main() -> None:
    settings = get_settings()
    import redis.asyncio as aioredis

    redis_client = aioredis.from_url(settings.redis_url, decode_responses=False)
    llm = make_router(settings, redis_client)

    async with GitHubClient(max_concurrency=3) as github_client:
        engine = make_engine()
        try:
            session_factory = make_session_factory(engine)
            context = ToolContext(
                github_client=github_client,
                session_factory=session_factory,
            )

            results: list[dict] = []
            for canary in CANARIES:
                print(f"→ Running canary: {canary.name}")
                trace = await run_canary(canary, context, llm)
                results.append(trace)

                # Write per-canary trace file
                provider_tag = os.environ.get("LLM_PROVIDER", "unknown")
                trace_path = TRACES_DIR / f"{provider_tag}__{canary.name}.json"
                trace_path.write_text(json.dumps(trace, indent=2, default=str))
                verdict = trace["verdict"]
                marker = "✓" if verdict.startswith("ok") else "✗"
                print(f"  {marker} {verdict}")
        finally:
            await engine.dispose()

    # Summary
    print("\n=== Canary results summary ===")
    ok_count = sum(1 for r in results if r["verdict"].startswith("ok"))
    fail_count = len(results) - ok_count
    print(f"  ok:        {ok_count}/{len(results)}")
    print(f"  failed:    {fail_count}/{len(results)}")
    print(f"\nTrace files written to: {TRACES_DIR}/")

    # Print failures
    if fail_count > 0:
        print("\n=== Failures ===")
        for r in results:
            if not r["verdict"].startswith("ok"):
                print(f"  {r['canary']['name']}: {r['verdict']}")


if __name__ == "__main__":
    asyncio.run(main())
