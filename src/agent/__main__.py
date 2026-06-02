from __future__ import annotations

import argparse
import asyncio

from api.db import make_engine, make_session_factory
from github_fetcher.client import GitHubClient

from agent.loop import run_agent
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
            result = await run_agent(
                prompt,
                tool_context=context,
                max_iterations=max_iterations,
            )
        finally:
            await engine.dispose()

    print("\n=== Agent summary ===")
    print(f"iterations:           {result.iterations}")
    print(f"hit_iteration_limit:  {result.hit_iteration_limit}")
    print(f"tool_calls ({len(result.tool_calls)}):")
    for name, args in result.tool_calls:
        print(f"  - {name}({args})")

    print("\n=== Final answer ===")
    print(result.text if result.text else "(no text response — hit iteration limit)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--max-iterations", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(amain(args.prompt, args.max_iterations))


if __name__ == "__main__":
    main()
