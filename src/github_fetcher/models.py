"""Pydantic models for the GitHub fetcher."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GitHubUser(BaseModel):
    """A GitHub user profile."""

    login: str
    name: str | None = None
    public_repos: int = Field(ge=0)
    followers: int = Field(ge=0)
    following: int = Field(ge=0)
    created_at: str  # ISO timestamp; keep as str for simplicity


class GitHubRepo(BaseModel):
    """A GitHub repository (subset of fields we care about)."""

    name: str
    full_name: str
    stargazers_count: int = Field(ge=0)
    forks_count: int = Field(ge=0)
    language: str | None = None
    fork: bool


class UserScore(BaseModel):
    """Computed developer score for a user."""

    login: str
    name: str | None
    total_stars: int
    total_forks: int
    public_repos: int
    followers: int
    score: float
