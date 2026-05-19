from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

app = FastAPI(title="Learning API", version="0.1.0")

_USERS: dict[int, dict[str, Any]] = {
    1: {"id": 1, "name": "Yogesh", "role": "engineer"},
    2: {"id": 2, "name": "Alice", "role": "manager"},
    3: {"id": 3, "name": "Bob", "role": "engineer"},
}


@app.get("/users/{user_id}")
async def get_user(user_id: int) -> dict[str, Any]:
    """Fetch a user by ID."""
    if user_id not in _USERS:
        raise HTTPException(status_code=404, detail=f"user {user_id} not found")
    return _USERS[user_id]


@app.get("/users")
async def list_users(
    role: str | None = Query(default=None, description="Filter by role"),
    limit: int = Query(default=10, ge=1, le=100),
) -> list[dict[str, Any]]:
    """List users, optionally filtered by role."""
    results = list(_USERS.values())
    if role is not None:
        results = [u for u in results if u["role"] == role]
    return results[:limit]


class CreateUserRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    role: str = Field(pattern=r"^(engineer|manager|designer)$")


class UserResponse(BaseModel):
    id: int
    name: str
    role: str


@app.post("/users", status_code=201, response_model=UserResponse)
async def create_user(payload: CreateUserRequest) -> UserResponse:
    """Create a new user."""
    new_id = max(_USERS.keys(), default=0) + 1
    user: dict[str, Any] = {"id": new_id, "name": payload.name, "role": payload.role}
    _USERS[new_id] = user
    return UserResponse(**user)
