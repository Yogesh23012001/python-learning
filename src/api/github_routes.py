"""HTTP routes for the GitHub fetcher service."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Request
from github_fetcher.client import GitHubClient, RateLimitError, UserNotFoundError
from github_fetcher.models import UserScore
from github_fetcher.service import score_many_users, score_user
from llm.router import LLMRouter
from llm.summarize import generate_developer_summary
from prometheus_client import Counter
from purgatory.domain.model import OpenedState
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from api.db import DbDep
from api.logging_config import get_logger
from api.model_orm import StoredScore
from api.telemetry import get_tracer


def get_llm_router(request: Request) -> LLMRouter:
    router: LLMRouter | None = getattr(request.app.state, "llm_client", None)
    if router is None:
        raise RuntimeError("llm_client not initialized")
    return router


LLMRouterDep = Annotated[LLMRouter, Depends(get_llm_router)]

# Wall-clock deadline for the score endpoint. Bounds total time across
# the live call + DB fallback + any retries — regardless of how generous
# the per-call httpx/sqlalchemy timeouts are.
SCORE_REQUEST_DEADLINE_S = 10.0

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
    db: DbDep,
    username: Annotated[str, Path(min_length=1, max_length=39, pattern=r"^[A-Za-z0-9-]+$")],
) -> UserScore:
    logger.info("scoring_user", username=username)
    try:
        async with asyncio.timeout(SCORE_REQUEST_DEADLINE_S):
            try:
                score = await score_user(client, username)
                scores_computed_total.labels(outcome="success").inc()
                logger.info("scoring_user_completed", username=username, score=score.score)
                return score
            except UserNotFoundError as e:
                scores_computed_total.labels(outcome="not_found").inc()
                raise HTTPException(status_code=404, detail=str(e)) from e
            except (RateLimitError, OpenedState, httpx.HTTPError) as e:
                # Live call failed. Try the stored fallback.
                logger.warning(
                    "score_user_live_failed_using_fallback",
                    username=username,
                    error=type(e).__name__,
                )
                stmt = select(StoredScore).where(StoredScore.login == username)
                result = await db.execute(stmt)
                stored = result.scalar_one_or_none()
                if stored is None:
                    scores_computed_total.labels(outcome="rate_limited").inc()
                    raise HTTPException(
                        status_code=503,
                        detail="upstream unavailable and no cached score",
                    ) from e
                scores_computed_total.labels(outcome="degraded").inc()
                return UserScore(
                    login=stored.login,
                    name=stored.name,
                    total_stars=stored.total_stars,
                    total_forks=stored.total_forks,
                    public_repos=stored.public_repos,
                    followers=stored.followers,
                    score=stored.score,
                )
    except TimeoutError as e:
        # Whole-request deadline exceeded — distinct from per-call timeouts,
        # which are caught above as httpx.HTTPError.
        scores_computed_total.labels(outcome="deadline_exceeded").inc()
        logger.error(
            "score_request_deadline_exceeded",
            username=username,
            deadline_s=SCORE_REQUEST_DEADLINE_S,
        )
        raise HTTPException(
            status_code=504,
            detail=f"request exceeded {SCORE_REQUEST_DEADLINE_S}s deadline",
        ) from e


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
    summary: str | None = None
    summary_generated_at: datetime | None = None
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


@router.post(
    "/users/{username}/summary",
    response_model=StoredScoreResponse,
)
async def generate_and_save_summary(
    client: ClientDep,
    db: DbDep,
    llm: LLMRouterDep,
    username: Annotated[
        str,
        Path(min_length=1, max_length=39, pattern=r"^[A-Za-z0-9-]+$"),
    ],
) -> StoredScoreResponse:
    """Score a user, generate an LLM summary, persist, and return."""
    logger.info("summary_request_received", username=username)

    # 1. Score the user (Week 1 Tuesday's logic)
    try:
        score = await score_user(client, username)
    except UserNotFoundError as e:
        logger.warning("summary_user_not_found", username=username)
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (RateLimitError, httpx.HTTPError) as e:
        # httpx.HTTPError covers ReadTimeout, ConnectError, NetworkError, and
        # HTTPStatusError. Unlike /score we don't have a DB fallback for new
        # summaries — the live call is required — so surface 503.
        logger.error(
            "summary_github_unavailable",
            username=username,
            error_type=type(e).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail=f"GitHub upstream unavailable ({type(e).__name__})",
        ) from e

    # 2. Generate the summary (Week 1 Friday's LLM work)
    try:
        summary_text = await generate_developer_summary(llm, user_score=score)
    except Exception as exc:
        logger.exception("summary_llm_failed", username=username)
        raise HTTPException(status_code=502, detail=f"summary generation failed: {exc}") from exc

    # 3. Upsert score + summary into Postgres (Week 1 Wednesday's persistence)
    now = datetime.now(UTC)
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
            summary=summary_text,
            summary_generated_at=now,
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
                "summary": summary_text,
                "summary_generated_at": now,
            },
        )
        .returning(StoredScore)
    )
    result = await db.execute(stmt)
    record = result.scalar_one()
    logger.info("summary_saved", username=username, summary_length=len(summary_text))
    return StoredScoreResponse.model_validate(record)
