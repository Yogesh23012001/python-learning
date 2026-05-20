"""Tests for the FastAPI app."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from api.app import _RATE_LIMIT_STORE, _score_cache, app
from api.github_routes import get_github_client
from fastapi.testclient import TestClient
from github_fetcher.client import GitHubClient, UserNotFoundError
from github_fetcher.models import GitHubRepo, GitHubUser


@pytest.fixture(autouse=True)
def reset_module_state() -> None:
    """Clear cross-test middleware state so tests don't pollute each other."""
    _RATE_LIMIT_STORE.clear()
    _score_cache.clear()


@pytest.fixture
def client() -> TestClient:
    """A fresh TestClient per test.

    NOTE: TestClient triggers the app's lifespan on __enter__/__exit__.
    Using it as a context manager ensures startup/shutdown run.
    """
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_root_returns_hello(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "hello"}


def test_root_includes_request_id_header(client: TestClient) -> None:
    response = client.get("/")
    assert "x-request-id" in response.headers
    assert "x-process-time-ms" in response.headers


def test_root_echoes_provided_request_id(client: TestClient) -> None:
    response = client.get("/", headers={"X-Request-ID": "test-trace-123"})
    assert response.headers["x-request-id"] == "test-trace-123"


def test_boom_returns_generic_500_envelope(client: TestClient) -> None:
    """Verify unhandled exceptions don't leak tracebacks."""
    response = client.get("/boom")
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "InternalServerError"
    assert body["detail"] == "an unexpected error occurred"
    assert "request_id" in body
    # Critical: no traceback, no class name, no file path
    assert "ValueError" not in str(body)
    assert "Traceback" not in str(body)


def test_payment_returns_402_with_business_error_envelope(client: TestClient) -> None:
    response = client.get("/payment")
    assert response.status_code == 402
    body = response.json()
    assert body["error"] == "PaymentDeclinedError"
    assert "insufficient funds" in body["detail"]
    assert "request_id" in body


def test_withdraw_402_when_insufficient_balance(client: TestClient) -> None:
    response = client.post("/withdraw", params={"amount": 100, "balance": 50})
    assert response.status_code == 402
    body = response.json()
    assert body["error"] == "InsufficientBalanceError"
    assert "50" in body["detail"] and "100" in body["detail"]


def test_withdraw_validation_error_returns_422(client: TestClient) -> None:
    # amount is a float — sending a string should 422
    response = client.post("/withdraw", params={"amount": "not-a-number"})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "ValidationError"
    assert isinstance(body["detail"], list)
    assert "request_id" in body


def make_fake_client(
    users: dict[str, GitHubUser],
    repos_by_user: dict[str, list[GitHubRepo]],
) -> MagicMock:
    """Build a MagicMock that quacks like a GitHubClient."""
    client = MagicMock(spec=GitHubClient)

    async def get_user(username: str) -> GitHubUser:
        if username not in users:
            raise UserNotFoundError(f"no such user: {username}")
        return users[username]

    async def get_user_repos(username: str, limit: int = 30) -> list[GitHubRepo]:
        if username not in users:
            raise UserNotFoundError(f"no such user: {username}")
        return repos_by_user.get(username, [])

    client.get_user = AsyncMock(side_effect=get_user)
    client.get_user_repos = AsyncMock(side_effect=get_user_repos)
    return client


@pytest.fixture
def fake_github_data() -> tuple[dict, dict]:
    users = {
        "alice": GitHubUser(
            login="alice",
            name="Alice",
            public_repos=2,
            followers=100,
            following=10,
            created_at="2020-01-01T00:00:00Z",
        ),
        "bob": GitHubUser(
            login="bob",
            name="Bob",
            public_repos=5,
            followers=50,
            following=20,
            created_at="2019-06-15T00:00:00Z",
        ),
    }
    repos = {
        "alice": [
            GitHubRepo(
                name="r1",
                full_name="alice/r1",
                stargazers_count=200,
                forks_count=20,
                language="Go",
                fork=False,
            ),
            GitHubRepo(
                name="r2",
                full_name="alice/r2",
                stargazers_count=50,
                forks_count=5,
                language="Python",
                fork=False,
            ),
        ],
        "bob": [
            GitHubRepo(
                name="bp",
                full_name="bob/bp",
                stargazers_count=10,
                forks_count=1,
                language="Rust",
                fork=False,
            ),
        ],
    }
    return users, repos


def test_get_user_score_returns_computed_score(
    client: TestClient,
    fake_github_data: tuple[dict, dict],
) -> None:
    users, repos = fake_github_data
    fake_client = make_fake_client(users, repos)

    # Override the dependency that resolves the GitHubClient
    app.dependency_overrides[get_github_client] = lambda: fake_client
    try:
        response = client.get("/github/users/alice/score")
        assert response.status_code == 200
        body = response.json()
        assert body["login"] == "alice"
        assert body["total_stars"] == 250  # 200 + 50
        assert body["total_forks"] == 25  # 20 + 5
        assert body["public_repos"] == 2
        assert body["followers"] == 100
        # score = 250*1 + 25*2 + 2*0.5 + 100*0.3 = 250 + 50 + 1 + 30 = 331
        assert body["score"] == 331.0
    finally:
        app.dependency_overrides.clear()


def test_get_user_score_404_when_not_found(
    client: TestClient,
    fake_github_data: tuple[dict, dict],
) -> None:
    users, repos = fake_github_data
    fake_client = make_fake_client(users, repos)
    app.dependency_overrides[get_github_client] = lambda: fake_client
    try:
        response = client.get("/github/users/ghost/score")
        assert response.status_code == 404
        assert "no such user" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_batch_endpoint_partitions_success_and_failure(
    client: TestClient,
    fake_github_data: tuple[dict, dict],
) -> None:
    users, repos = fake_github_data
    fake_client = make_fake_client(users, repos)
    app.dependency_overrides[get_github_client] = lambda: fake_client
    try:
        response = client.post(
            "/github/users/score-batch",
            json={"usernames": ["alice", "ghost", "bob"]},
        )
        assert response.status_code == 200
        items = response.json()["items"]
        assert len(items) == 3
        # Order preserved
        assert items[0]["username"] == "alice"
        assert items[0]["score"] is not None
        assert items[0]["error"] is None

        assert items[1]["username"] == "ghost"
        assert items[1]["score"] is None
        assert items[1]["error"] == "not_found"

        assert items[2]["username"] == "bob"
        assert items[2]["score"] is not None
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def override_github_client(fake_github_data: tuple[dict, dict]):
    users, repos = fake_github_data
    fake_client = make_fake_client(users, repos)
    app.dependency_overrides[get_github_client] = lambda: fake_client
    yield fake_client  # tests get access to the fake
    app.dependency_overrides.clear()  # always cleans up


def test_score_endpoint_calls_client_correctly(
    client: TestClient,
    override_github_client: MagicMock,
) -> None:
    response = client.get("/github/users/alice/score")
    assert response.status_code == 200

    # The fake_client's methods are AsyncMock — they record calls
    override_github_client.get_user.assert_awaited_with("alice")
    override_github_client.get_user_repos.assert_awaited_with("alice")


def test_rate_limit_blocks_request_over_quota(client: TestClient) -> None:
    """Send (limit + 1) requests; everything up to `limit` passes, then 429."""
    from api.config import get_settings

    limit = get_settings().rate_limit_max_requests
    for _ in range(limit):
        response = client.get("/")
        assert response.status_code == 200
    response = client.get("/")
    assert response.status_code == 429
    assert response.json()["error"] == "TooManyRequests"
