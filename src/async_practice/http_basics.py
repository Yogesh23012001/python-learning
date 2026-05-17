"""Hour 2 — async HTTP basics with httpx."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
# )

# logger = logging.getLogger(__name__)


# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=1, max=10),
#     retry=retry_if_exception_type(
#         (httpx.TimeoutException, httpx.NetworkError)
#     ),
#     before_sleep=before_sleep_log(logger, logging.WARNING),
#     reraise=True,
# )


async def fetch_post(client: httpx.AsyncClient, post_id: int) -> dict[str, Any]:
    response = await client.get(f"https://jsonplaceholder.typicode.com/posts/{post_id}")
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    return data


async def fetch_with_semaphore(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    post_id: int,
) -> dict[str, Any]:
    async with sem:
        return await fetch_post(client, post_id)


async def benchmark_concurrency(limit: int, ids: list[int]) -> float:
    """Run fetch with a semaphore of given size, return elapsed seconds."""
    sem = asyncio.Semaphore(limit)
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        start = time.perf_counter()
        await asyncio.gather(*(fetch_with_semaphore(sem, client, i) for i in ids))
        return time.perf_counter() - start


async def main() -> None:
    ids = list(range(1, 100))  # 99 posts

    for limit in [1, 5, 20, 50, 100]:
        elapsed = await benchmark_concurrency(limit, ids)
        print(f"concurrency={limit:>2}: {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
