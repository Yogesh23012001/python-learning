# python-learning

A 12-week roadmap project — building production-grade Python services from zero to AI infrastructure.

## Week 1 — Foundations & First LLM Integration

### What's in here

- `src/url_shortener/` — typed Python URL shortener with TTL and click tracking (Mon)
- `src/github_fetcher/` — async GitHub client with retries, semaphores, hexagonal architecture (Tue)
- `src/api/` — FastAPI service: middleware, exception handlers, Postgres + Alembic, structlog, OpenTelemetry, Prometheus, Grafana, circuit breakers, graceful degradation (Wed-Thu)
- `src/llm/` — multi-provider LLM abstraction with Gemini + Mock providers, structured outputs, streaming, prompt caching (Fri)

### Highlights

- 100% test coverage on business logic; mypy --strict; ruff clean
- Three-pillar observability (logs, metrics, traces) with Grafana dashboards
- Provider-agnostic LLM router (`LLMProvider` protocol + adapters)
- Real combined feature: `POST /github/users/{username}/summary` — fetches GitHub data, generates LLM summary, persists, exposes with full tracing

### Running locally

```bash
docker compose up -d
uv run alembic upgrade head
uv run uvicorn api.app:app --reload --app-dir src --port 8000
```

- API docs: http://localhost:8000/docs
- Jaeger: http://localhost:16686
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000