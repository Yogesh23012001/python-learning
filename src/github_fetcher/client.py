"""HTTP client for the GitHub API.

Encapsulates: connection pooling, timeouts, retries, rate limiting,
structured logging. Knows nothing about scoring or business logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from purgatory import AsyncCircuitBreakerFactory
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from github_fetcher.models import GitHubRepo, GitHubUser

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=2.0)
DEFAULT_MAX_CONCURRENCY = 5


class GitHubAPIError(Exception):
    """Base exception for GitHub API errors."""


class UserNotFoundError(GitHubAPIError):
    """The requested GitHub user does not exist."""


class RateLimitError(GitHubAPIError):
    """GitHub API rate limit exceeded."""


class GitHubClient:
    """Async client for the GitHub REST API.

    Designed to be created once per service lifetime (long-lived).
    Use as an async context manager to ensure connection cleanup.
    """

    def __init__(
        self,
        *,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        max_retries: int = 3,
        breaker_threshold: int = 5,  # 5 failures → open
        breaker_ttl: float = 30.0,  # stay open for 30s, then probe
    ) -> None:
        self._client: httpx.AsyncClient | None = None
        self._timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries
        self._breakers = AsyncCircuitBreakerFactory(
            default_threshold=breaker_threshold,
            default_ttl=breaker_ttl,
        )

    async def __aenter__(self) -> GitHubClient:
        self._client = httpx.AsyncClient(
            base_url=GITHUB_API_BASE,
            timeout=self._timeout,
            headers={"Accept": "application/vnd.github+json"},
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Internal request helper — single source of HTTP truth
    # ------------------------------------------------------------------

    async def _request(self, path: str) -> dict[str, Any] | list[Any]:
        if self._client is None:
            raise RuntimeError("client used outside `async with` context")

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),  # type: ignore[arg-type, unused-ignore]
            reraise=True,
        ):
            with attempt:
                async with self._semaphore:  # noqa: SIM117
                    async with await self._breakers.get_breaker("github"):
                        response = await self._client.get(path)

                if response.status_code == 404:
                    raise UserNotFoundError(f"resource not found: {path}")
                if response.status_code == 403:
                    raise RateLimitError("github api rate limit exceeded")
                response.raise_for_status()
                data: dict[str, Any] | list[Any] = response.json()
                return data

        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_user(self, username: str) -> GitHubUser:
        """Fetch a GitHub user's profile."""
        data = await self._request(f"/users/{username}")
        assert isinstance(data, dict)
        return GitHubUser.model_validate(data)

    async def get_user_repos(self, username: str, limit: int = 30) -> list[GitHubRepo]:
        """Fetch a user's top repos by stars."""
        data = await self._request(f"/users/{username}/repos?sort=updated&per_page={limit}")
        assert isinstance(data, list)
        return [GitHubRepo.model_validate(r) for r in data]
