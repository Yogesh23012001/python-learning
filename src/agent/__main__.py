"""Run the agent from the command line for quick testing.

Usage:
    uv run python -m agent "what time is it?"
"""

from __future__ import annotations

import asyncio
import sys

from agent.loop import run_single_tool_call


async def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m agent <prompt>")
        sys.exit(1)
    prompt = " ".join(sys.argv[1:])
    result = await run_single_tool_call(prompt)
    print("\n=== Final answer ===")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
