"""Tests for the /calc/batch endpoint in api.practice."""

from __future__ import annotations

import pytest
from api.practice import app
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_batch_with_mixed_ops_returns_correct_results(client: TestClient) -> None:
    """Successful batch with mixed ops returns correct results."""
    response = client.post(
        "/calc/batch",
        json={
            "operations": [
                {"op": "add", "a": 2, "b": 3},
                {"op": "subtract", "a": 10, "b": 4},
                {"op": "multiply", "a": 6, "b": 7},
                {"op": "divide", "a": 20, "b": 5},
            ]
        },
    )
    assert response.status_code == 200
    assert response.json() == {"results": [5.0, 6.0, 42.0, 4.0]}


def test_batch_divide_by_zero_returns_null_not_500(client: TestClient) -> None:
    """Divide-by-zero in batch returns null for that op, not a 500."""
    response = client.post(
        "/calc/batch",
        json={
            "operations": [
                {"op": "add", "a": 2, "b": 2},
                {"op": "divide", "a": 5, "b": 0},
                {"op": "multiply", "a": 3, "b": 4},
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["results"] == [4.0, None, 12.0]


def test_batch_empty_operations_returns_422(client: TestClient) -> None:
    """Empty operations list returns 422 (min_length=1 constraint)."""
    response = client.post("/calc/batch", json={"operations": []})
    assert response.status_code == 422


def test_batch_too_many_operations_returns_422(client: TestClient) -> None:
    """Operations list longer than 100 returns 422."""
    ops = [{"op": "add", "a": 1, "b": 1} for _ in range(101)]
    response = client.post("/calc/batch", json={"operations": ops})
    assert response.status_code == 422
