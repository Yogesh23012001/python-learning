"""CLI entry point for the GitHub fetcher."""

from __future__ import annotations

import asyncio
import logging
import sys

from github_fetcher.client import GitHubClient, UserNotFoundError
from github_fetcher.models import UserScore
from github_fetcher.service import score_many_users


async def main(usernames: list[str]) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    async with GitHubClient() as client:
        results = await score_many_users(client, usernames)

    print("\n=== Scores ===")
    for username, result in zip(usernames, results, strict=True):
        if isinstance(result, UserScore):
            print(
                f"  {result.login} ({result.name or '—'}): "
                f"score={result.score} "
                f"stars={result.total_stars} forks={result.total_forks} "
                f"repos={result.public_repos} followers={result.followers}"
            )
        elif isinstance(result, UserNotFoundError):
            print(f"  {username}: NOT FOUND")
        else:
            print(f"  {username}: ERROR ({type(result).__name__}: {result})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m github_fetcher.cli <username> [<username> ...]")
        sys.exit(1)
    asyncio.run(main(sys.argv[1:]))
