"""Prometheus metrics setup."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.responses import Response as StarletteResponse

# RED metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=("method", "path", "status"),
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=("method", "path"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

http_requests_in_flight = Gauge(
    "http_requests_in_flight",
    "HTTP requests currently being processed",
)

agent_runs_total = Counter(
    "agent_runs_total",
    "Total agent runs by outcome",
    labelnames=("outcome",),  # "completed" | "max_iterations"
)

agent_iterations = Histogram(
    "agent_iterations",
    "Number of iterations per agent run",
    buckets=(1, 2, 3, 4, 5, 6, 8, 10, 15),
)

agent_tool_calls_total = Counter(
    "agent_tool_calls_total",
    "Total tool calls by tool name and outcome",
    labelnames=("tool", "outcome"),
)

agent_cost_usd = Histogram(
    "agent_cost_usd",
    "Cost in USD per agent run",
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)


def add_metrics_middleware(app: FastAPI) -> None:
    """Add a middleware that records RED metrics for every request."""

    @app.middleware("http")
    async def record_metrics(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path == "/metrics":
            return await call_next(request)

        http_requests_in_flight.inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            raise
        finally:
            duration = time.perf_counter() - start
            route = request.scope.get("route")
            path = route.path if route is not None else request.url.path
            http_requests_total.labels(
                method=request.method,
                path=path,
                status=str(status),
            ).inc()
            http_request_duration_seconds.labels(
                method=request.method,
                path=path,
            ).observe(duration)
            http_requests_in_flight.dec()  # ← always decrement on exit

        return response


db_pool_size = Gauge(
    "db_pool_size",
    "SQLAlchemy connection pool size",
    labelnames=("state",),  # "checked_in" or "checked_out"
)


def collect_db_pool_metrics(engine: AsyncEngine) -> None:
    """Snapshot the engine's pool state into Prometheus gauges."""
    sync_pool = engine.sync_engine.pool
    # NullPool, used in tests, doesn't expose checkedout/checkedin
    if hasattr(sync_pool, "checkedout"):
        db_pool_size.labels(state="checked_out").set(sync_pool.checkedout())
    if hasattr(sync_pool, "checkedin"):
        db_pool_size.labels(state="checked_in").set(sync_pool.checkedin())


def add_metrics_endpoint(app: FastAPI) -> None:
    """Expose /metrics for Prometheus scraping."""

    @app.get("/metrics", include_in_schema=False)
    async def metrics(request: Request) -> StarletteResponse:
        # Snapshot pool state at scrape time
        engine = getattr(request.app.state, "engine", None)
        if engine is not None:
            collect_db_pool_metrics(engine)
        data = generate_latest(REGISTRY)
        return StarletteResponse(content=data, media_type=CONTENT_TYPE_LATEST)
