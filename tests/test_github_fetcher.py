"""Tests for the GitHub fetcher."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from github_fetcher.client import GitHubClient, UserNotFoundError
from github_fetcher.models import GitHubRepo, GitHubUser, UserScore
from github_fetcher.service import _compute_score, score_user
from pydantic import ValidationError

# ===================================================================
# Pure logic tests — no mocks needed
# ===================================================================


def test_compute_score_basic() -> None:
    score = _compute_score(
        total_stars=100,
        total_forks=10,
        public_repos=20,
        followers=50,
    )
    # 100 + 20 + 10 + 15 = 145
    assert score == 145.0


def test_compute_score_all_zero() -> None:
    assert _compute_score(total_stars=0, total_forks=0, public_repos=0, followers=0) == 0.0


# ===================================================================
# Pydantic tests
# ===================================================================


def test_github_user_validates_required_fields() -> None:
    u = GitHubUser(
        login="yogesh",
        public_repos=5,
        followers=10,
        following=2,
        created_at="2020-01-01T00:00:00Z",
    )
    assert u.login == "yogesh"
    assert u.name is None


def test_github_user_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        GitHubUser(
            login="yogesh",
            public_repos=-1,
            followers=10,
            following=2,
            created_at="2020-01-01T00:00:00Z",
        )


# ===================================================================
# Service tests with mocked client
# ===================================================================


@pytest.mark.asyncio
async def test_score_user_aggregates_correctly() -> None:
    # Build a mocked GitHubClient
    mock_client = MagicMock(spec=GitHubClient)
    mock_client.get_user = AsyncMock(
        return_value=GitHubUser(
            login="testuser",
            name="Test User",
            public_repos=10,
            followers=100,
            following=20,
            created_at="2020-01-01T00:00:00Z",
        )
    )
    mock_client.get_user_repos = AsyncMock(
        return_value=[
            GitHubRepo(
                name="r1",
                full_name="t/r1",
                stargazers_count=50,
                forks_count=5,
                language="Go",
                fork=False,
            ),
            GitHubRepo(
                name="r2",
                full_name="t/r2",
                stargazers_count=30,
                forks_count=2,
                language="Python",
                fork=False,
            ),
            GitHubRepo(
                name="forked",
                full_name="other/x",
                stargazers_count=1000,
                forks_count=100,
                language="Go",
                fork=True,
            ),  # ignored
        ]
    )

    result = await score_user(mock_client, "testuser")

    assert isinstance(result, UserScore)
    assert result.login == "testuser"
    assert result.total_stars == 80  # 50 + 30, fork ignored
    assert result.total_forks == 7  # 5 + 2, fork ignored
    assert result.public_repos == 10
    assert result.followers == 100


@pytest.mark.asyncio
async def test_score_user_propagates_not_found() -> None:
    mock_client = MagicMock(spec=GitHubClient)
    mock_client.get_user = AsyncMock(side_effect=UserNotFoundError("nope"))
    mock_client.get_user_repos = AsyncMock(side_effect=UserNotFoundError("nope"))

    with pytest.raises(UserNotFoundError):
        await score_user(mock_client, "ghost")
