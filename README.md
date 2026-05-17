# python-learning

A learning-oriented Python project covering modern packaging, type-checked
business logic, async I/O, and a strict toolchain (ruff + mypy + pytest + uv).

The repo is organized as three small but realistic packages under `src/`,
each focused on a distinct skill.

---

## Project layout

```
myproject/
├── src/
│   ├── url_shortener/      # Sync service: URL shortener with TTL & aliases
│   ├── github_fetcher/     # Async HTTP client + scoring for the GitHub API
│   └── async_practice/     # Exercises: httpx, asyncio, pydantic
├── tests/                  # pytest suites for each package
├── pyproject.toml          # uv / hatch / ruff / mypy / pytest config
└── uv.lock
```

---

## Packages

### `url_shortener`
A sync URL-shortening service. Supports auto-generated codes, custom aliases,
expiry (`ttl_seconds`), click tracking, top-N lookup, and a deterministic
clock injection point for testing.

Key exceptions: `CodeAlreadyExistsError`, `CodeNotFoundError`,
`CodeExpiredError`, `InvalidAliasError`, `GenerationFailedError`.

### `github_fetcher`
An async client for the GitHub REST API built on `httpx` and `tenacity`.

- Connection pooling and bounded concurrency via `asyncio.Semaphore`
- Retry with exponential backoff on transient errors
- Status-code-aware error classification (`UserNotFoundError`, `RateLimitError`)
- A `score_user` / `score_many_users` service layer that fans out concurrently
- A `python -m github_fetcher.cli <user> ...` CLI entrypoint

### `async_practice`
Small exercises supporting the rest of the project: `asyncio` patterns,
basic `httpx` use, and a `pydantic` intro.

---

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency and
environment management.

```bash
# 1) Install dependencies (creates .venv automatically)
uv sync

# 2) Activate the venv (optional; uv run handles this for you)
source .venv/bin/activate
```

Python 3.12+ is required (see `.python-version`).

---

## Running the tools

```bash
# Run the full test suite with coverage
uv run pytest

# Lint
uv run ruff check src tests

# Auto-format
uv run ruff format src tests

# Strict type-checking on a package
uv run mypy src/github_fetcher
uv run mypy src/url_shortener
```

Pytest is pre-configured (in `pyproject.toml`) with:
- `--cov=url_shortener --cov=github_fetcher --cov=async_practice`
- `--cov-report=term-missing`
- `asyncio_mode = "auto"` for the async tests

---

## Running the GitHub fetcher CLI

```bash
uv run python -m github_fetcher.cli torvalds gvanrossum
```

Outputs a per-user score line, or `NOT FOUND` / `ERROR` for failed lookups.
Failures from one user do not abort the others — `asyncio.gather` is run
with `return_exceptions=True`.

---

## Conventions enforced

- **Strict typing** (`strict = true` in `pyproject.toml`).
- **Ruff** lint + format with these rule sets: `E, W, F, I, B, UP, N, SIM, RUF`.
- **Line length** 100, double-quoted strings, Python 3.12 target.
- **Src layout** — packages live under `src/`, installed as a package by
  `uv` (`package = true`) and built by Hatchling.

---

## License

MIT — see [LICENSE](LICENSE).
