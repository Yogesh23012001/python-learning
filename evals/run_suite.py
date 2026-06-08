"""Run the full eval suite, aggregate scores, write a run report."""

from __future__ import annotations

import asyncio
import json
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.events import AgentCompleted, AgentStopped, ToolRequested
from agent.loop import run_agent_stream
from agent.tools import ToolContext
from anthropic import AsyncAnthropic
from api.config import get_settings
from api.db import make_engine, make_session_factory
from github_fetcher.client import GitHubClient
from llm.factory import make_router
from llm.router import LLMRouter

from evals.graders.deterministic import grade_deterministic
from evals.graders.judge import grade_with_judge

EVALS_DIR = Path(__file__).parent
RUNS_DIR = EVALS_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

# Cheap judge for fast iteration. Flip to "claude-sonnet-4-5" before committing
# a baseline so the reference numbers aren't biased by self-grading.
JUDGE_MODEL = "claude-haiku-4-5-20251001"


async def run_one(
    example: dict[str, Any],
    *,
    llm: LLMRouter,
    context: ToolContext,
    judge_client: AsyncAnthropic,
) -> dict[str, Any]:
    """Run the agent on one example, grade with both graders, return a result row."""
    actual_tools: list[str] = []
    terminal: AgentCompleted | AgentStopped | None = None

    try:
        async for event in run_agent_stream(
            example["prompt"],
            llm=llm,
            tool_context=context,
        ):
            if isinstance(event, ToolRequested):
                actual_tools.append(event.name)
            if isinstance(event, AgentCompleted | AgentStopped):
                terminal = event
    except Exception as exc:
        return {
            "id": example["id"],
            "category": example["category"],
            "agent_crashed": True,
            "error": str(exc)[:300],
            "deterministic_score": 0.0,
            "judge_score": 0.0,
        }

    actual_text = terminal.text if isinstance(terminal, AgentCompleted) else ""
    actual_outcome = terminal.reason if isinstance(terminal, AgentStopped) else "completed"

    det_score = grade_deterministic(
        example=example,
        actual_tools=actual_tools,
        actual_text=actual_text,
        actual_outcome=actual_outcome,
    )

    judge_score = await grade_with_judge(
        client=judge_client,
        example=example,
        actual_tools=actual_tools,
        actual_text=actual_text,
        judge_model=JUDGE_MODEL,
    )

    return {
        "id": example["id"],
        "category": example["category"],
        "prompt": example["prompt"],
        "actual_tools": actual_tools,
        "actual_text": actual_text[:500],  # truncate for run file
        "actual_outcome": actual_outcome,
        "deterministic": {
            "score": det_score.score,
            "tool_selection_ok": det_score.tool_selection_ok,
            "must_contain_ok": det_score.must_contain_ok,
            "must_not_contain_ok": det_score.must_not_contain_ok,
            "notes": det_score.notes,
        },
        "judge": {
            "score": judge_score.score,
            "correctness": judge_score.correctness,
            "appropriateness": judge_score.appropriateness,
            "reasoning": judge_score.reasoning,
            "error": judge_score.error,
        },
        "agent_crashed": False,
    }


async def main() -> None:
    # Load dataset
    dataset_path = EVALS_DIR / "datasets.jsonl"
    examples = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(examples)} examples")

    settings = get_settings()
    if settings.anthropic_api_key is None:
        raise SystemExit("ANTHROPIC_API_KEY not set in .env — required for judge grader")
    llm = make_router(settings)
    judge_client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())

    results: list[dict[str, Any]] = []

    async with GitHubClient(max_concurrency=3) as github_client:
        engine = make_engine()
        try:
            session_factory = make_session_factory(engine)
            context = ToolContext(
                github_client=github_client,
                session_factory=session_factory,
            )

            for i, example in enumerate(examples, 1):
                print(
                    f"[{i}/{len(examples)}] {example['id']} ({example['category']})...",
                    end=" ",
                    flush=True,
                )
                result = await run_one(example, llm=llm, context=context, judge_client=judge_client)
                results.append(result)
                if result["agent_crashed"]:
                    print(f"CRASH: {result['error'][:80]}")
                else:
                    det = result["deterministic"]["score"]
                    judge = result["judge"]["score"]
                    print(f"det={det:.2f}, judge={judge:.2f}")
        finally:
            await engine.dispose()

    # Aggregate
    summary = _aggregate(results)

    # Write run file — derive metadata from actual config so swaps don't lie
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_path = RUNS_DIR / f"{timestamp}_baseline.json"
    run_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "provider": llm.provider_name,
                "model": llm.default_model,
                "judge_model": JUDGE_MODEL,
                "n_examples": len(examples),
                "summary": summary,
                "results": results,
            },
            indent=2,
            default=str,
        )
    )

    # Print summary
    print()
    print("=" * 60)
    print(f"Eval run complete. Written to: {run_path.name}")
    print("=" * 60)
    print(f"Overall det score:   {summary['overall']['deterministic_mean']:.2f}")
    print(f"Overall judge score: {summary['overall']['judge_mean']:.2f}")
    print(f"Crash rate:          {summary['overall']['crash_rate']:.0%}")
    print()
    print("Per-category breakdown:")
    print(f"{'category':<28} {'n':>3} {'det':>6} {'judge':>6}")
    for cat, stats in sorted(summary["by_category"].items()):
        print(
            f"{cat:<28} {stats['n']:>3} {stats['deterministic_mean']:>6.2f} {stats['judge_mean']:>6.2f}"
        )


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build per-category and overall aggregates."""
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    cat_summary = {}
    for cat, rows in by_cat.items():
        det_scores = [r["deterministic"]["score"] for r in rows if not r["agent_crashed"]]
        judge_scores = [r["judge"]["score"] for r in rows if not r["agent_crashed"]]
        crashes = sum(1 for r in rows if r["agent_crashed"])
        cat_summary[cat] = {
            "n": len(rows),
            "crashes": crashes,
            "deterministic_mean": statistics.mean(det_scores) if det_scores else 0.0,
            "judge_mean": statistics.mean(judge_scores) if judge_scores else 0.0,
        }

    all_det = [r["deterministic"]["score"] for r in results if not r["agent_crashed"]]
    all_judge = [r["judge"]["score"] for r in results if not r["agent_crashed"]]
    overall = {
        "deterministic_mean": statistics.mean(all_det) if all_det else 0.0,
        "judge_mean": statistics.mean(all_judge) if all_judge else 0.0,
        "crash_rate": sum(1 for r in results if r["agent_crashed"]) / len(results),
    }

    return {
        "overall": overall,
        "by_category": cat_summary,
    }


if __name__ == "__main__":
    asyncio.run(main())
