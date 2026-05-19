"""Production-shape FastAPI app."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from github_fetcher.client import GitHubClient

from api.db import make_engine, make_session_factory
from api.github_routes import router as github_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("starting up — initializing resources")

    # Database
    app.state.engine = make_engine()
    app.state.session_factory = make_session_factory(app.state.engine)

    # GitHub client (existing)
    app.state.github_client = GitHubClient(max_concurrency=5)
    await app.state.github_client.__aenter__()

    # In-memory state (existing)
    app.state.fake_llm = {"calls_made": 0}

    yield

    logger.info("shutting down — cleaning up resources")
    await app.state.github_client.__aexit__(None, None, None)
    await app.state.engine.dispose()  # close all pool connections
    logger.info("final call count: %d", app.state.fake_llm["calls_made"])


app = FastAPI(title="Production-Shape API", version="0.1.0", lifespan=lifespan)
app.include_router(github_router)


# ============================================================
# 5-minute response cache for GET /github/users/{username}/score
# ============================================================

_USER_SCORE_PATH = re.compile(r"^/github/users/(?P<username>[A-Za-z0-9-]+)/score$")
_SCORE_CACHE_TTL_SECONDS = 300
# username -> (cached_at, body_bytes, media_type)
_score_cache: dict[str, tuple[float, bytes, str]] = {}


@app.middleware("http")
async def cache_user_scores(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Cache 200 responses for the user-score endpoint for 5 minutes."""
    if request.method != "GET":
        return await call_next(request)

    match = _USER_SCORE_PATH.match(request.url.path)
    if match is None:
        return await call_next(request)

    username = match.group("username")
    now = time.time()

    entry = _score_cache.get(username)
    if entry is not None and now - entry[0] < _SCORE_CACHE_TTL_SECONDS:
        logger.info("cache hit username=%s", username)
        return Response(
            content=entry[1],
            media_type=entry[2],
            headers={"X-Cache": "HIT"},
        )

    # BaseHTTPMiddleware actually returns a StreamingResponse at runtime,
    # though the static type is Response (which has no body_iterator). Cast
    # so mypy lets us drain the body for caching.
    from typing import cast

    from starlette.responses import StreamingResponse

    response = await call_next(request)
    if response.status_code != 200:
        return response
    streaming = cast(StreamingResponse, response)

    # Drain the body iterator — it can only be consumed once, so we
    # cache the bytes and rebuild a fresh Response for the client.
    body_chunks: list[bytes] = []
    async for chunk in streaming.body_iterator:
        if isinstance(chunk, bytes):
            body_chunks.append(chunk)
        elif isinstance(chunk, str):
            body_chunks.append(chunk.encode())
        else:  # memoryview
            body_chunks.append(bytes(chunk))
    body = b"".join(body_chunks)

    media_type = response.media_type or "application/json"
    _score_cache[username] = (now, body, media_type)

    return Response(
        content=body,
        status_code=response.status_code,
        media_type=media_type,
        headers={**dict(response.headers), "X-Cache": "MISS"},
    )


@app.middleware("http")
async def add_request_id_and_timing(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Attach a request_id to every request and log timing."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.perf_counter()

    # Make request_id available to downstream handlers via request.state
    request.state.request_id = request_id

    try:
        response = await call_next(request)
    except Exception:
        # Re-raise — the exception handler will deal with the response
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "request failed: method=%s path=%s elapsed_ms=%.2f request_id=%s",
            request.method,
            request.url.path,
            elapsed_ms,
            request_id,
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

    logger.info(
        "method=%s path=%s status=%d elapsed_ms=%.2f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "hello"}


@app.get("/slow")
async def slow() -> dict[str, str]:
    import asyncio

    await asyncio.sleep(0.5)
    return {"done": "yes"}


@app.get("/boom")
async def boom() -> dict[str, str]:
    raise ValueError("something exploded")


class BusinessError(Exception):
    """Base class for business-logic errors that should be exposed to clients."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class PaymentDeclinedError(BusinessError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"payment declined: {reason}", status_code=402)


@app.exception_handler(BusinessError)
async def business_error_handler(
    request: Request,
    exc: BusinessError,
) -> JSONResponse:
    """Convert business errors into standard JSON responses."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(
        "business_error: type=%s message=%s request_id=%s",
        type(exc).__name__,
        exc.message,
        request_id,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": type(exc).__name__,
            "detail": exc.message,
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Catch-all for unexpected errors. Never leak the traceback to clients."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(
        "unhandled_exception: type=%s request_id=%s",
        type(exc).__name__,
        request_id,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "an unexpected error occurred",
            "request_id": request_id,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Customize Pydantic validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "detail": exc.errors(),
            "request_id": request_id,
        },
    )


@app.get("/payment")
async def process_payment() -> dict[str, str]:
    raise PaymentDeclinedError("insufficient funds")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_RATE_LIMIT_STORE: dict[str, list[float]] = {}  # client_ip -> list of request timestamps


@app.middleware("http")
async def rate_limit(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Simple in-memory rate limiter (for demonstration only)."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Clean up old entries
    window = 60  # seconds
    _RATE_LIMIT_STORE[client_ip] = [
        timestamp for timestamp in _RATE_LIMIT_STORE.get(client_ip, []) if now - timestamp < window
    ]

    if len(_RATE_LIMIT_STORE[client_ip]) >= 10:  # limit to 10 requests per minute
        return JSONResponse(
            status_code=429,
            content={"error": "TooManyRequests", "detail": "rate limit exceeded"},
        )

    _RATE_LIMIT_STORE.setdefault(client_ip, []).append(now)
    return await call_next(request)


@app.post("/chat")
async def chat() -> dict[str, int]:
    app.state.fake_llm["calls_made"] += 1
    return {"calls_so_far": app.state.fake_llm["calls_made"]}


class InsufficientBalanceError(BusinessError):
    def __init__(self, balance: float, requested: float) -> None:
        super().__init__(
            f"balance ${balance} < requested ${requested}",
            status_code=402,
        )


@app.post("/withdraw")
async def withdraw(amount: float = 100, balance: float = 50) -> dict[str, float]:
    if amount > balance:
        raise InsufficientBalanceError(balance=balance, requested=amount)
    return {"withdrawn": amount, "remaining_balance": balance - amount}
