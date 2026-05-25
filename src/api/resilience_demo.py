"""Hour 5 — timeout patterns."""

from __future__ import annotations

import asyncio


async def slow_op(duration: float) -> str:
    await asyncio.sleep(duration)
    return f"finished after {duration}s"


async def main() -> None:
    # 1. Per-call timeout — only this call is bounded
    try:
        async with asyncio.timeout(0.5):
            result = await slow_op(2.0)
            print(result)
    except TimeoutError:
        print("per-call timeout fired")

    # 2. Whole-block deadline — all child awaits share the budget
    try:
        async with asyncio.timeout(1.0):
            s = await slow_op(0.4)
            print(s)
            t = await slow_op(0.4)
            print(t)
            u = await slow_op(0.4)
            print(u)  # this exceeds the 1.0s budget
    except TimeoutError:
        print("budget exhausted across multiple awaits")


if __name__ == "__main__":
    asyncio.run(main())
