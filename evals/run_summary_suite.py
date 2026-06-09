"""Run the summary suite across legacy, agentic-no-extra-tool, and agentic-with-fetch-top-repos.

For each user in summary_dataset.jsonl, call all three endpoints and score the
output. Write a comparison JSON.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from anthropic import AsyncAnthropic
from api.config import get_settings

from evals.graders.summary_grader import grade_summary

EVALS_DIR = Path(__file__).parent
RUNS_DIR = EVALS_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)


SUMMARY_BASE_URL = "http://localhost:8000"


async def fetch_summary(
    client: httpx.AsyncClient,
    login: str,
    endpoint: str,
) -> tuple[str, dict[str, Any] | None]:
    """Call the summary endpoint, return (summary_text, profile_data_or_none)."""
    url = f"{SUMMARY_BASE_URL}/github/users/{login}/summary{endpoint}"
    try:
        response = await client.post(url, timeout=60.0)
        response.raise_for_status()
        body = response.json()
        return body.get("summary", ""), body
    except Exception as exc:
        return f"[ERROR: {exc!s}]", None


async def main() -> None:
    dataset_path = EVALS_DIR / "summary_dataset.jsonl"
    examples = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(examples)} summary examples")

    settings = get_settings()
    if settings.anthropic_api_key is None:
        raise SystemExit("ANTHROPIC_API_KEY not set in .env")
    judge_client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())

    variants = [
        ("legacy", ""),
        ("agentic_no_top_repos", "/agentic"),  # See note below
        ("agentic_with_top_repos", "/agentic"),  # See note below
    ]

    # Note: we can't have BOTH agentic variants simultaneously without flag-gating.
    # For this hour, choose: run the agentic endpoint as it currently is (which
    # has the new tool registered) AND compare to legacy. If you want to compare
    # WITH and WITHOUT the new tool, you'd need to temporarily disable the tool
    # registration. We'll comment this limitation in the verdict doc.

    # Reduce to two variants for the actual run
    variants = [
        ("legacy", ""),
        ("agentic_with_top_repos", "/agentic"),
    ]

    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient() as http_client:
        for example in examples:
            login = example["login"]
            print(f"\n[{example['id']}] {login} ({example['profile_type']})")
            example_results = {"id": example["id"], "login": login, "variants": {}}

            for variant_name, endpoint in variants:
                print(f"  - {variant_name}...", end=" ", flush=True)
                summary_text, profile = await fetch_summary(http_client, login, endpoint)

                if not profile or not summary_text or summary_text.startswith("[ERROR"):
                    example_results["variants"][variant_name] = {
                        "summary": summary_text[:200],
                        "score": None,
                        "error": "endpoint failed or returned empty",
                    }
                    print("FAILED")
                    continue

                # Build ground-truth data for the judge
                ground_truth = {
                    "login": profile.get("login"),
                    "name": profile.get("name"),
                    "total_stars": profile.get("total_stars"),
                    "total_forks": profile.get("total_forks"),
                    "public_repos": profile.get("public_repos"),
                    "followers": profile.get("followers"),
                    "score": profile.get("score"),
                }

                score = await grade_summary(
                    client=judge_client,
                    login=login,
                    summary_text=summary_text,
                    actual_profile_data=ground_truth,
                )

                example_results["variants"][variant_name] = {
                    "summary": summary_text,
                    "specificity": score.specificity,
                    "factual_grounding": score.factual_grounding,
                    "hallucination_risk": score.hallucination_risk,
                    "readability": score.readability,
                    "overall": score.overall,
                    "judge_reasoning": score.judge_reasoning,
                    "error": score.error,
                }
                print(f"overall={score.overall:.2f}")

            results.append(example_results)

    # Write run file
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_path = RUNS_DIR / f"{timestamp}_summary_comparison.json"
    run_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "n_examples": len(examples),
                "variants": [name for name, _ in variants],
                "results": results,
            },
            indent=2,
            default=str,
        )
    )

    # Print summary
    print("\n" + "=" * 70)
    print(f"Summary suite complete. Written to: {run_path.name}")
    print("=" * 70)

    for variant_name, _ in variants:
        scores = [
            r["variants"][variant_name]["overall"]
            for r in results
            if variant_name in r["variants"]
            and r["variants"][variant_name].get("overall") is not None
        ]
        if scores:
            mean = sum(scores) / len(scores)
            print(f"{variant_name:30s} n={len(scores)} overall={mean:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
