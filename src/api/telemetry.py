"""OpenTelemetry tracing setup."""

from __future__ import annotations

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from sqlalchemy.ext.asyncio import AsyncEngine


def configure_tracing(
    *,
    service_name: str,
    otlp_endpoint: str = "http://localhost:4317",
) -> None:
    """Set up OpenTelemetry to export traces to Jaeger via OTLP."""
    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)


def instrument_fastapi_and_httpx(app: FastAPI) -> None:
    """Instrument the FastAPI app and httpx BEFORE any client is constructed.

    Call this at module load, right after `app = FastAPI(...)` is created.
    HTTPXClientInstrumentor patches httpx.Client.__init__ globally, so it must
    run before any httpx client is instantiated (e.g. our GitHubClient).
    """
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()


def instrument_sqlalchemy(engine: AsyncEngine) -> None:
    """Instrument SQLAlchemy. Call this after the engine is created."""
    SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)


def get_tracer(name: str) -> trace.Tracer:
    """Get a named tracer for emitting custom spans."""
    return trace.get_tracer(name)
