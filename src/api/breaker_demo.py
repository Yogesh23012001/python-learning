"""Demo: watch the circuit breaker trip and reset."""

from __future__ import annotations

import asyncio

from github_fetcher.client import GitHubClient


async def main() -> None:
    # Use an URL that returns 5xx — httpbin's /status endpoint is perfect
    async with GitHubClient(breaker_threshold=3, breaker_ttl=10.0) as client:
        # Manually override the base URL via the underlying httpx client
        client._client.base_url = "https://httpbin.org"

        for i in range(8):
            try:
                # Forces a 500 response
                await client._request("/status/500")
                print(f"attempt {i + 1}: success")
            except Exception as e:
                print(f"attempt {i + 1}: {type(e).__name__}: {e}")
            await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())
