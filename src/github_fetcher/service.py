"""Business logic for scoring GitHub users."""

from __future__ import annotations

import asyncio
import logging

from github_fetcher.client import GitHubClient
from github_fetcher.models import UserScore

logger = logging.getLogger(__name__)


def _compute_score(
    *,
    total_stars: int,
    total_forks: int,
    public_repos: int,
    followers: int,
) -> float:
    """Heuristic developer score. Tune as needed."""
    return total_stars * 1.0 + total_forks * 2.0 + public_repos * 0.5 + followers * 0.3


async def score_user(client: GitHubClient, username: str) -> UserScore:
    """Fetch a user + their repos, compute and return their score."""
    user_task = client.get_user(username)
    repos_task = client.get_user_repos(username)
    user, repos = await asyncio.gather(user_task, repos_task)

    own_repos = [r for r in repos if not r.fork]
    total_stars = sum(r.stargazers_count for r in own_repos)
    total_forks = sum(r.forks_count for r in own_repos)

    score = _compute_score(
        total_stars=total_stars,
        total_forks=total_forks,
        public_repos=user.public_repos,
        followers=user.followers,
    )

    return UserScore(
        login=user.login,
        name=user.name,
        total_stars=total_stars,
        total_forks=total_forks,
        public_repos=user.public_repos,
        followers=user.followers,
        score=round(score, 2),
    )


async def score_many_users(
    client: GitHubClient,
    usernames: list[str],
) -> list[UserScore | BaseException]:
    """Score multiple users concurrently.

    Returns successes as UserScore, failures as the raised exception.
    Caller decides how to handle each.
    """
    return await asyncio.gather(
        *(score_user(client, u) for u in usernames),
        return_exceptions=True,
    )
