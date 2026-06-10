"""Production-shape FastAPI app."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
import structlog
from agent.agent_routes import router as agent_router
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from github_fetcher.client import GitHubClient
from llm.factory import make_router
from redis.exceptions import RedisError

from api.config import SettingsDep, get_settings
from api.db import make_engine, make_session_factory
from api.github_routes import router as github_router
from api.llm_routes import router as llm_router
from api.logging_config import configure_logging, get_logger
from api.mertics import add_metrics_endpoint, add_metrics_middleware
from api.telemetry import (
    configure_tracing,
    instrument_fastapi_and_httpx,
    instrument_sqlalchemy,
)

_settings = get_settings()
configure_logging(
    json_logs=_settings.log_format_json,
    level=getattr(logging, _settings.log_level.value),
)
logger = get_logger(__name__)

# Configure tracing at module load — before any httpx client or DB engine is
# created. HTTPXClientInstrumentor patches httpx.Client.__init__ globally and
# only affects clients constructed AFTER it runs.
configure_tracing(service_name="python-learning-api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info(
        "starting_up",
        environment=settings.environment.value,
        debug=settings.debug,
    )

    app.state.settings = settings
    app.state.engine = make_engine()
    instrument_sqlalchemy(app.state.engine)
    app.state.session_factory = make_session_factory(app.state.engine)
    app.state.github_client = GitHubClient(
        max_concurrency=settings.github_max_concurrency,
        timeout=httpx.Timeout(
            connect=5.0,
            read=settings.github_api_timeout_s,
            write=5.0,
            pool=2.0,
        ),
    )

    # Redis client for the LLM prompt cache. Ping at startup so a misconfigured
    # URL fails loud — but keep the LLM call path resilient: cache get/set is
    # wrapped in try/except RedisError so a runtime outage degrades to misses.
    app.state.redis_client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False,
    )
    try:
        await app.state.redis_client.ping()
        logger.info("redis_connected", url=settings.redis_url)
    except RedisError as exc:
        logger.warning(
            "redis_unavailable_at_startup",
            url=settings.redis_url,
            error=str(exc)[:200],
        )

    if settings.gemini_api_key is None:
        raise RuntimeError("GEMINI_API_KEY not set in .env")
    app.state.llm_client = make_router(settings, app.state.redis_client)
    await app.state.github_client.__aenter__()
    app.state.fake_llm = {"calls_made": 0}

    yield

    logger.info("shutting_down")
    await app.state.github_client.__aexit__(None, None, None)
    await app.state.redis_client.aclose()
    await app.state.engine.dispose()


app = FastAPI(title="Production-Shape API", version="0.1.0", lifespan=lifespan)

# Instrument BEFORE custom middleware decorators run, so OTel's middleware
# sits at the outermost layer and creates the parent span for each request.
instrument_fastapi_and_httpx(app)

app.include_router(github_router)
app.include_router(llm_router)
app.include_router(agent_router)


# /metrics endpoint can be registered now; it's a route, not middleware.
add_metrics_endpoint(app)
# NOTE: add_metrics_middleware(app) is intentionally NOT called here.
# It's called at the bottom of this file so the metrics middleware ends up
# OUTERMOST in the stack — otherwise cache hits short-circuit before
# reaching metrics and you under-count requests.


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
    """Attach request_id to context; emit structured request log on completion."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start = time.perf_counter()
    request.state.request_id = request_id

    # Bind context for the duration of this request
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception("request_failed", elapsed_ms=round(elapsed_ms, 2))
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

    logger.info(
        "request_completed",
        status=response.status_code,
        elapsed_ms=round(elapsed_ms, 2),
    )
    return response


# Register metrics middleware LAST so it sits at the outermost layer of the
# stack and observes every incoming request — including those served from the
# in-process cache before reaching the route handler.
add_metrics_middleware(app)


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
    """Catch-all for unexpected errors. Never leak the traceback to clients.

    Re-raises framework exceptions (HTTPException, RequestValidationError) so
    their dedicated handlers can run — otherwise this swallows them as 500s.
    """
    from fastapi import HTTPException

    if isinstance(exc, HTTPException | RequestValidationError):
        raise exc
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
    """Customize Pydantic validation errors.

    Pydantic v2's `errors()` puts the raw `ValueError` exception object inside
    `ctx["error"]` for custom `@field_validator` failures — not JSON-serializable.
    Strip `ctx` so JSON encoding always succeeds.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    detail = [{k: v for k, v in err.items() if k != "ctx"} for err in exc.errors()]
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "detail": detail,
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
    _settings = get_settings()
    window = _settings.rate_limit_window_seconds
    rate_limit_max = _settings.rate_limit_max_requests
    _RATE_LIMIT_STORE[client_ip] = [
        timestamp for timestamp in _RATE_LIMIT_STORE.get(client_ip, []) if now - timestamp < window
    ]

    if (
        len(_RATE_LIMIT_STORE[client_ip]) >= rate_limit_max
    ):  # limit to specified requests per minute
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


@app.get("/info")
async def info(settings: SettingsDep) -> dict[str, str | int]:
    return {
        "environment": settings.environment.value,
        "log_level": settings.log_level.value,
        "github_max_concurrency": settings.github_max_concurrency,
        "rate_limit_max_requests": settings.rate_limit_max_requests,
    }
