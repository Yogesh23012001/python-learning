"""Run one example end-to-end through the agent + graders."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent.events import AgentCompleted, AgentStopped, ToolRequested
from agent.loop import run_agent_stream
from agent.tools import ToolContext
from anthropic import AsyncAnthropic
from api.config import get_settings
from api.db import make_engine, make_session_factory
from github_fetcher.client import GitHubClient
from llm.factory import make_router

from evals.graders.deterministic import grade_deterministic
from evals.graders.judge import grade_with_judge


async def main(example_id: str) -> None:
    # Load dataset
    dataset_path = Path(__file__).parent / "datasets.jsonl"
    examples = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    example = next((e for e in examples if e["id"] == example_id), None)
    if example is None:
        print(f"Example {example_id} not found")
        return

    print(f"=== Running example: {example['id']} ===")
    print(f"Prompt: {example['prompt']}")
    print(f"Category: {example['category']}")
    print()

    settings = get_settings()
    llm = make_router(settings)

    # Run the agent
    async with GitHubClient(max_concurrency=3) as github_client:
        engine = make_engine()
        try:
            session_factory = make_session_factory(engine)
            context = ToolContext(
                github_client=github_client,
                session_factory=session_factory,
            )

            actual_tools: list[str] = []
            terminal: AgentCompleted | AgentStopped | None = None
            async for event in run_agent_stream(
                example["prompt"],
                llm=llm,
                tool_context=context,
            ):
                if isinstance(event, ToolRequested):
                    actual_tools.append(event.name)
                if isinstance(event, AgentCompleted | AgentStopped):
                    terminal = event
        finally:
            await engine.dispose()

    actual_text = terminal.text if isinstance(terminal, AgentCompleted) else ""
    actual_outcome = terminal.reason if isinstance(terminal, AgentStopped) else "completed"

    print(f"Tools called: {actual_tools}")
    print(f"Outcome: {actual_outcome}")
    print(f"Response: {actual_text[:200]}")
    print()

    # Grade deterministically
    det_score = grade_deterministic(
        example=example,
        actual_tools=actual_tools,
        actual_text=actual_text,
        actual_outcome=actual_outcome,
    )
    print("=== Deterministic grade ===")
    print(f"Score: {det_score.score:.2f}")
    print(f"Tool selection: {det_score.tool_selection_ok}")
    print(f"Must contain: {det_score.must_contain_ok}")
    print(f"Must not contain: {det_score.must_not_contain_ok}")
    if det_score.notes:
        for note in det_score.notes:
            print(f"  - {note}")
    print()

    # Grade with judge — pull key from settings (which reads .env)
    if settings.anthropic_api_key is None:
        raise SystemExit("ANTHROPIC_API_KEY not set in .env")
    judge_client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    judge_score = await grade_with_judge(
        client=judge_client,
        example=example,
        actual_tools=actual_tools,
        actual_text=actual_text,
    )
    print("=== Judge grade ===")
    print(f"Score: {judge_score.score:.2f}")
    print(f"Correctness: {judge_score.correctness}/5")
    print(f"Appropriateness: {judge_score.appropriateness}/5")
    print(f"Reasoning: {judge_score.reasoning}")
    if judge_score.error:
        print(f"  Error: {judge_score.error}")


if __name__ == "__main__":
    import sys

    example_id = sys.argv[1] if len(sys.argv) > 1 else "tool_select_001"
    asyncio.run(main(example_id))
