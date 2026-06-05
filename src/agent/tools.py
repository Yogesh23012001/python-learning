"""Tool definitions, schemas, and dispatch for the agent system.

A "tool" is a Python function the LLM can request the system to call. Each
tool has three parts:
  1. The implementation (a regular Python function)
  2. A JSON schema describing it to the LLM
  3. Registration in HANDLERS for dispatch by name

The pattern is identical to the worker handler registry in
idempotent-task-queue — same engineering shape, different domain.
"""

from __future__ import annotations

import ast
import asyncio
import operator
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from api.model_orm import StoredScore
from github_fetcher.client import GitHubClient, RateLimitError, UserNotFoundError
from github_fetcher.service import score_user
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# Forward-declared types to avoid circular imports

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ToolContext:
    """Resources tools may need. Injected by the agent runner."""

    github_client: GitHubClient | None = None
    session_factory: async_sessionmaker[AsyncSession] | None = None


class ToolRefusedError(Exception):
    """Raised when a tool is refused by policy."""


# Denylist — tools that NEVER run regardless of LLM request
REFUSED_TOOLS: set[str] = set()  # populated by add_refused_tool()


def add_refused_tool(name: str) -> None:
    """Register a tool name that is forbidden to execute."""
    REFUSED_TOOLS.add(name)
    logger.info("tool_refusal_registered", tool=name)


# ============================================================
# Types
# ============================================================

ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[dict[str, Any]]]

# ============================================================
# Tool implementations
# ============================================================

_ALLOWED_OPS: dict[type[ast.AST], Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Tiny safe arithmetic evaluator — no function calls, no names."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


async def calculate(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    expression = args.get("expression")
    if not expression or not isinstance(expression, str):
        return {"error": "missing 'expression' (string)"}
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except Exception as exc:
        return {"error": f"could not evaluate: {exc}"}
    return {"expression": expression, "result": result}


async def get_current_time(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Return the current UTC time as ISO 8601."""
    now = datetime.now(UTC).isoformat()
    return {"utc_now": now}


async def lookup_github_user(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Fetch a GitHub user's profile and computed score."""
    login = args.get("login")
    if not login or not isinstance(login, str):
        return {"error": "missing 'login' (string) argument"}
    if not login.replace("-", "").isalnum() or len(login) > 39:
        return {"error": f"invalid github login: {login!r}"}

    if context.github_client is None:
        return {"error": "github_client not configured"}

    try:
        score = await score_user(context.github_client, login)
    except UserNotFoundError:
        return {"error": "user_not_found", "login": login}
    except RateLimitError:
        return {"error": "github_rate_limit"}

    return {
        "login": score.login,
        "name": score.name,
        "total_stars": score.total_stars,
        "total_forks": score.total_forks,
        "public_repos": score.public_repos,
        "followers": score.followers,
        "score": score.score,
    }


async def score_and_save_user(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Score a GitHub user AND persist (upsert) the result to our database."""
    login = args.get("login")
    if not login or not isinstance(login, str):
        return {"error": "missing 'login' (string) argument"}
    if not login.replace("-", "").isalnum() or len(login) > 39:
        return {"error": f"invalid github login: {login!r}"}

    if context.github_client is None:
        return {"error": "github_client not configured"}
    if context.session_factory is None:
        return {"error": "session_factory not configured"}

    try:
        score = await score_user(context.github_client, login)
    except UserNotFoundError:
        return {"error": "user_not_found", "login": login}
    except RateLimitError:
        return {"error": "github_rate_limit"}

    from sqlalchemy.dialects.postgresql import insert as pg_insert

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
    )
    async with context.session_factory() as session:
        await session.execute(stmt)
        await session.commit()

    return {
        "saved": True,
        "login": score.login,
        "name": score.name,
        "score": score.score,
    }


async def query_stored_scores(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """List GitHub scores from our database, ordered by score descending."""
    limit = args.get("limit", 10)
    if not isinstance(limit, int) or limit < 1 or limit > 50:
        return {"error": "limit must be an integer between 1 and 50"}

    if context.session_factory is None:
        return {"error": "session_factory not configured"}

    async with context.session_factory() as session:
        stmt = select(StoredScore).order_by(desc(StoredScore.score)).limit(limit)
        result = await session.execute(stmt)
        rows = list(result.scalars().all())

    return {
        "count": len(rows),
        "scores": [
            {
                "login": r.login,
                "name": r.name,
                "score": r.score,
                "total_stars": r.total_stars,
                "public_repos": r.public_repos,
            }
            for r in rows
        ],
    }


async def web_search(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Stub web search — returns fake but plausible results.

    In production this would call Bing, Brave Search, Tavily, or similar.
    For Week 2 it returns a fixed shape so we can study agent behavior
    without paying for real search APIs.
    """
    query = args.get("query")
    if not query or not isinstance(query, str):
        return {"error": "missing 'query' (string) argument"}

    # Synthetic but plausible-looking results
    # Real implementation would hit https://api.bing.microsoft.com/... or similar
    return {
        "query": query,
        "results": [
            {
                "title": f"Article about {query} — example.com",
                "url": f"https://example.com/articles/{query.replace(' ', '-')}",
                "snippet": (
                    f"This is a stub result for the query '{query}'. "
                    "In production, this would be a real web result."
                ),
            },
            {
                "title": f"{query.title()} — Wikipedia",
                "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                "snippet": (f"Wikipedia article summary about {query} (stub data)."),
            },
            {
                "title": f"Top discussion about {query} on Hacker News",
                "url": "https://news.ycombinator.com/item?id=123456",
                "snippet": f"Community discussion of {query} (stub data).",
            },
        ],
    }


async def summarize_text(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Summarize a passage of text using the LLM.

    Architectural note: this is an agent tool that itself calls an LLM.
    Nested LLM calls are real in production — 'summarize this output',
    'classify this result', 'translate this snippet' are all this pattern.
    """
    text = args.get("text")
    max_sentences = args.get("max_sentences", 2)

    if not text or not isinstance(text, str):
        return {"error": "missing 'text' (string) argument"}
    if not isinstance(max_sentences, int) or max_sentences < 1 or max_sentences > 10:
        return {"error": "max_sentences must be int 1-10"}
    if len(text) > 5000:
        return {"error": "text too long (max 5000 chars)"}

    # Inline LLM call — we go direct to Gemini rather than through the agent's
    # LLMRouter, to avoid circular dependencies and keep this tool simple.
    from api.config import get_settings
    from google import genai
    from google.genai import types as gtypes

    settings = get_settings()
    if settings.gemini_api_key is None:
        return {"error": "GEMINI_API_KEY not configured"}
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    prompt = (
        f"Summarize the following text in at most {max_sentences} "
        f"complete sentences. Be concise and factual.\n\n{text}"
    )

    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=gtypes.GenerateContentConfig(max_output_tokens=300),
        )
        summary = response.text or ""
    except Exception as exc:
        return {"error": f"summarization_failed: {exc}"}

    return {
        "original_length": len(text),
        "summary": summary.strip(),
        "max_sentences": max_sentences,
    }


# ============================================================
# Tool schemas (described to the LLM)
# ============================================================


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_current_time",
        "description": (
            "Return the current UTC time as an ISO 8601 string. "
            "Use when the user asks about the current time, today's date, "
            "or any question that requires knowing 'now'."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "lookup_github_user",
        "description": (
            "Fetch a GitHub user's public profile data and a computed developer "
            "score (based on stars, forks, repos, followers). "
            "Use when the user asks about a specific GitHub username, "
            "developer profile, or wants to compare GitHub users."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "login": {
                    "type": "string",
                    "description": "The GitHub username (login), 1-39 chars, alphanumeric and hyphens.",
                },
            },
            "required": ["login"],
        },
    },
    {
        "name": "query_stored_scores",
        "description": (
            "Retrieve GitHub user scores previously saved in our database, "
            "ranked from highest to lowest. "
            "Use when the user asks 'who are our top users', 'show me saved scores', "
            "'list the developers in our database', or similar phrases referring to "
            "stored or previously-scored users."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-50, default 10).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a math expression with +, -, *, /, **, % operators. "
            "Use whenever exact arithmetic is needed — comparing two numbers, "
            "computing ratios, percentages, sums. LLMs make arithmetic mistakes; "
            "delegate to this tool for any non-trivial calculation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression to evaluate, e.g. '462000 / 84000' or '(245 + 63) * 2'.",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for general-knowledge information not in our database. "
            "Returns the top 3 search results with title, URL, and snippet. "
            "Use when the user asks about real-world topics that aren't about "
            "specific GitHub users, our stored data, or time/math — for example, "
            "questions about historical events, current news, technology concepts, "
            "or company information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, 1-200 characters.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "summarize_text",
        "description": (
            "Summarize a passage of text in 1-10 sentences using an LLM. "
            "Use when the user provides a long text and asks for a summary, "
            "TL;DR, or shortened version. Also useful when another tool returns "
            "verbose data that needs to be condensed for the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to summarize (max 5000 characters).",
                },
                "max_sentences": {
                    "type": "integer",
                    "description": "Maximum number of sentences in the summary (1-10, default 2).",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "score_and_save_user",
        "description": (
            "Look up a GitHub user AND persist their score to our database "
            "for future reference. Use ONLY when the user explicitly asks to "
            "save, record, or remember a developer — for example: 'add this user "
            "to our database', 'save torvalds for later', 'remember this developer'. "
            "Do NOT use for simple lookups — use lookup_github_user instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "login": {
                    "type": "string",
                    "description": "The GitHub username, 1-39 chars.",
                },
            },
            "required": ["login"],
        },
    },
]


# ============================================================
# Registry + dispatch
# ============================================================


HANDLERS: dict[str, ToolHandler] = {
    "get_current_time": get_current_time,
    "lookup_github_user": lookup_github_user,
    "score_and_save_user": score_and_save_user,
    "query_stored_scores": query_stored_scores,
    "calculate": calculate,
    "web_search": web_search,
    "summarize_text": summarize_text,
}

DEFAULT_TOOL_TIMEOUT_S = 15.0


class ToolNotFoundError(Exception):
    """Raised when the LLM requests a tool we don't have."""


class ToolTimeoutError(Exception):
    """Raised when a tool exceeds its execution timeout."""


async def execute_tool(
    name: str,
    args: dict[str, Any],
    context: ToolContext,
    *,
    timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
) -> dict[str, Any]:
    """Dispatch a tool call with a wall-clock timeout."""
    if name in REFUSED_TOOLS:
        logger.warning("tool_call_refused", tool=name, args=args)
        raise ToolRefusedError(f"tool {name} is denied by policy")
    handler = HANDLERS.get(name)
    if handler is None:
        raise ToolNotFoundError(f"no tool registered with name={name!r}")

    logger.info("tool_call_started", tool=name, args=args)
    try:
        async with asyncio.timeout(timeout_s):
            result = await handler(args, context)
    except TimeoutError as exc:
        logger.warning("tool_call_timeout", tool=name, timeout_s=timeout_s)
        raise ToolTimeoutError(f"tool {name} exceeded {timeout_s}s") from exc
    except Exception as exc:
        logger.warning("tool_call_failed", tool=name, error=str(exc)[:200])
        raise
    logger.info("tool_call_completed", tool=name, result_keys=list(result.keys()))
    return result
