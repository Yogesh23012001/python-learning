"""HTTP routes for the GitHub fetcher service."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from github_fetcher.client import GitHubClient, RateLimitError, UserNotFoundError
from github_fetcher.models import UserScore
from github_fetcher.service import score_many_users, score_user
from prometheus_client import Counter
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from api.db import DbDep
from api.logging_config import get_logger
from api.model_orm import StoredScore
from api.telemetry import get_tracer

tracer = get_tracer(__name__)

logger = get_logger(__name__)

router = APIRouter(prefix="/github", tags=["github"])


def get_github_client(request: Request) -> GitHubClient:
    """Dependency that retrieves the shared GitHubClient from app state."""
    client: GitHubClient | None = getattr(request.app.state, "github_client", None)
    if client is None:
        raise RuntimeError("github_client not initialized — check lifespan")
    return client


ClientDep = Annotated[GitHubClient, Depends(get_github_client)]

scores_computed_total = Counter(
    "scores_computed_total",
    "GitHub user scores computed",
    labelnames=("outcome",),  # "success" | "not_found" | "rate_limited"
)


# ============================================================
# Request/response models
# ============================================================


class BatchScoreRequest(BaseModel):
    usernames: list[str] = Field(min_length=1, max_length=20)


class BatchScoreItem(BaseModel):
    username: str
    score: UserScore | None = None
    error: str | None = None


class BatchScoreResponse(BaseModel):
    items: list[BatchScoreItem]


# ============================================================
# Routes
# ============================================================


@router.get("/users/{username}/score", response_model=UserScore)
async def get_user_score(
    client: ClientDep,
    username: Annotated[str, Path(min_length=1, max_length=39, pattern=r"^[A-Za-z0-9-]+$")],
) -> UserScore:
    """Compute and return a developer score for a GitHub user."""
    logger.info("scoring_user", username=username)
    try:
        score = await score_user(client, username)
    except UserNotFoundError as e:
        scores_computed_total.labels(outcome="not_found").inc()
        logger.warning("score_user_not_found", username=username)
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RateLimitError as e:
        scores_computed_total.labels(outcome="rate_limited").inc()
        logger.error("github_rate_limited", username=username)
        raise HTTPException(status_code=503, detail=str(e)) from e

    scores_computed_total.labels(outcome="success").inc()
    logger.info("scoring_user_completed", username=username, score=score.score)
    return score


@router.post("/users/score-batch", response_model=BatchScoreResponse)
async def score_users_batch(
    client: ClientDep,
    payload: BatchScoreRequest,
) -> BatchScoreResponse:
    """Score multiple GitHub users concurrently."""
    logger.info("scoring_users_batch", usernames=payload.usernames)
    results = await score_many_users(client, payload.usernames)
    items: list[BatchScoreItem] = []
    for username, result in zip(payload.usernames, results, strict=True):
        if isinstance(result, UserScore):
            items.append(BatchScoreItem(username=username, score=result))
        elif isinstance(result, UserNotFoundError):
            items.append(BatchScoreItem(username=username, error="not_found"))
        else:
            items.append(BatchScoreItem(username=username, error=type(result).__name__))
    return BatchScoreResponse(items=items)


class StoredScoreResponse(BaseModel):
    id: int
    login: str
    name: str | None
    total_stars: int
    total_forks: int
    public_repos: int
    followers: int
    score: float
    created_at: datetime

    model_config = {"from_attributes": True}  # allow ORM → Pydantic conversion


@router.post("/users/{username}/score-and-save", response_model=StoredScoreResponse)
async def score_and_save(
    client: ClientDep,
    db: DbDep,
    username: Annotated[str, Path(min_length=1, max_length=39, pattern=r"^[A-Za-z0-9-]+$")],
) -> StoredScoreResponse:
    with tracer.start_as_current_span("score_and_save") as span:
        span.set_attribute("user.login", username)
        try:
            score = await score_user(client, username)
        except UserNotFoundError as e:
            span.set_attribute("error.type", "not_found")
            logger.warning("score_user_not_found", username=username)
            raise HTTPException(status_code=404, detail=str(e)) from e
        except RateLimitError as e:
            span.set_attribute("error.type", "rate_limited")
            logger.error("github_rate_limited", username=username)
            raise HTTPException(status_code=503, detail=str(e)) from e

        span.set_attribute("score.value", score.score)
        span.set_attribute("score.public_repos", score.public_repos)

        stmt = (
            pg_insert(StoredScore)
            .values(
                login=score.login,
                name=score.name,
                total_stars=score.total_stars,
                total_forks=score.total_forks,
                public_repos=score.public_repos,
                followers=score.followers,
                score=score.score,
            )
            .on_conflict_do_update(
                index_elements=["login"],
                set_={
                    "name": score.name,
                    "total_stars": score.total_stars,
                    "total_forks": score.total_forks,
                    "public_repos": score.public_repos,
                    "followers": score.followers,
                    "score": score.score,
                },
            )
            .returning(StoredScore)
        )
        result = await db.execute(stmt)
        return StoredScoreResponse.model_validate(result.scalar_one())


@router.get("/users/{username}/stored", response_model=StoredScoreResponse)
async def get_stored_score(
    db: DbDep,
    username: Annotated[str, Path(min_length=1, max_length=39, pattern=r"^[A-Za-z0-9-]+$")],
) -> StoredScoreResponse:
    """Retrieve a previously-saved score from the DB."""
    logger.info("retrieving_stored_score", username=username)
    stmt = select(StoredScore).where(StoredScore.login == username)
    result = await db.execute(stmt)
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail=f"no stored score for {username}")
    logger.info("retrieved_stored_score", username=username, score=record.score)
    return StoredScoreResponse.model_validate(record)


@router.get("/stored-scores", response_model=list[StoredScoreResponse])
async def list_stored_scores(db: DbDep) -> list[StoredScoreResponse]:
    """List all stored scores, newest first."""
    logger.info("listing_stored_scores")
    stmt = select(StoredScore).order_by(StoredScore.created_at.desc())
    result = await db.execute(stmt)
    logger.info("listed_stored_scores", count=len(result.scalars().all()))  # ← exhausts the cursor
    return [
        StoredScoreResponse.model_validate(record) for record in result.scalars().all()
    ]  # ← empty!
