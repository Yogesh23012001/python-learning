"""Run the agent from the command line — events stream live.

Usage:
    uv run python -m agent "your prompt here"
    uv run python -m agent "your prompt here" --max-iterations 5
"""

from __future__ import annotations

import argparse
import asyncio

from api.db import make_engine, make_session_factory
from github_fetcher.client import GitHubClient

from agent.events import (
    AgentCompleted,
    AgentStarted,
    AgentStopped,
    IterationStarted,
    ToolCompleted,
    ToolFailed,
    ToolRequested,
)
from agent.loop import run_agent_stream
from agent.tools import ToolContext


async def amain(prompt: str, max_iterations: int) -> None:
    async with GitHubClient(max_concurrency=3) as github_client:
        engine = make_engine()
        try:
            session_factory = make_session_factory(engine)
            context = ToolContext(
                github_client=github_client,
                session_factory=session_factory,
            )

            final_text = ""
            terminal_reason = "completed"
            total_cost = 0.0
            total_iterations = 0

            async for event in run_agent_stream(
                prompt,
                tool_context=context,
                max_iterations=max_iterations,
            ):
                if isinstance(event, AgentStarted):
                    print(f"\n→ Agent started (model={event.model})")
                elif isinstance(event, IterationStarted):
                    print(f"\n  Iteration {event.iteration}")
                elif isinstance(event, ToolRequested):
                    print(f"  ⚙️  → {event.name}({event.args})")
                elif isinstance(event, ToolCompleted):
                    print(f"  ✓  ← {event.name} ({event.duration_ms:.0f}ms)")
                elif isinstance(event, ToolFailed):
                    print(f"  ✗  ← {event.name} FAILED: {event.error_type} — {event.error[:80]}")
                elif isinstance(event, AgentCompleted):
                    final_text = event.text
                    total_cost = event.total_cost_usd
                    total_iterations = event.iterations
                elif isinstance(event, AgentStopped):
                    terminal_reason = event.reason
                    total_cost = event.total_cost_usd
                    total_iterations = event.iterations
        finally:
            await engine.dispose()

    print(
        f"\n=== Done (reason: {terminal_reason}, iterations: {total_iterations}, cost: ${total_cost:.6f}) ==="
    )
    print(final_text or "(no text response)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--max-iterations", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(amain(args.prompt, args.max_iterations))


if __name__ == "__main__":
    main()
