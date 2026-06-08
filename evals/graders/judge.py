"""LLM-as-judge grader — uses Claude to score response quality."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import structlog
from anthropic import AsyncAnthropic

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class JudgeScore:
    """Result of LLM-as-judge grading."""

    score: float  # 0.0 - 1.0
    reasoning: str  # judge's explanation
    correctness: int  # 1-5 sub-score: did the agent give a correct answer?
    appropriateness: int  # 1-5 sub-score: was the approach reasonable?
    error: str | None = None


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of AI agent responses. Your job is to grade an agent's response to a user prompt against the following criteria:

1. CORRECTNESS (1-5): Did the agent produce a correct, factually accurate answer?
   - 5: Perfectly correct
   - 4: Mostly correct, minor errors
   - 3: Partially correct, mixed signal
   - 2: Mostly incorrect
   - 1: Completely wrong or hallucinated

2. APPROPRIATENESS (1-5): Was the approach reasonable for the user's request?
   - 5: Ideal approach, used the right tools and reasoning
   - 4: Good approach with minor inefficiency
   - 3: Workable but suboptimal
   - 2: Wrong approach, even if final answer was lucky
   - 1: Completely wrong approach

You will respond with ONLY a JSON object in this exact format:
{
  "correctness": <int 1-5>,
  "appropriateness": <int 1-5>,
  "reasoning": "<2-3 sentence explanation>"
}

Do NOT include any text outside the JSON. Do NOT use markdown code blocks. Just the JSON object."""


def _judge_user_prompt(
    *,
    user_prompt: str,
    expected_behavior: str,
    actual_tools: list[str],
    actual_text: str,
) -> str:
    return f"""USER PROMPT:
{user_prompt}

EXPECTED BEHAVIOR:
{expected_behavior}

AGENT'S TOOL CALLS:
{json.dumps(actual_tools)}

AGENT'S FINAL RESPONSE:
{actual_text}

Grade this response. Output ONLY the JSON object."""


async def grade_with_judge(
    *,
    client: AsyncAnthropic,
    example: dict[str, Any],
    actual_tools: list[str],
    actual_text: str,
    judge_model: str = "claude-sonnet-4-5",
) -> JudgeScore:
    """Send one example + response to the judge LLM for grading.

    Use a stronger model as judge than the model being evaluated.
    Haiku-judging-Haiku is a known weakness.
    """
    expected_behavior = example.get("notes", "(no expected behavior described)")

    judge_prompt = _judge_user_prompt(
        user_prompt=example["prompt"],
        expected_behavior=expected_behavior,
        actual_tools=actual_tools,
        actual_text=actual_text,
    )

    try:
        response = await client.messages.create(
            model=judge_model,
            max_tokens=300,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": judge_prompt}],
        )
    except Exception as exc:
        return JudgeScore(
            score=0.0,
            reasoning="judge call failed",
            correctness=0,
            appropriateness=0,
            error=str(exc)[:200],
        )

    # Extract text content
    text = "".join(b.text for b in response.content if b.type == "text").strip()

    # Parse JSON
    try:
        # Sometimes Claude wraps in code blocks despite instructions; strip them
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        parsed = json.loads(text)
    except (json.JSONDecodeError, IndexError) as exc:
        return JudgeScore(
            score=0.0,
            reasoning=f"failed to parse judge output: {text[:200]}",
            correctness=0,
            appropriateness=0,
            error=str(exc)[:200],
        )

    correctness = int(parsed.get("correctness", 0))
    appropriateness = int(parsed.get("appropriateness", 0))
    reasoning = parsed.get("reasoning", "")

    # Score = average of the two 1-5 ratings, normalized to 0-1
    avg_rating = (correctness + appropriateness) / 2.0
    score = (avg_rating - 1.0) / 4.0  # 1 → 0.0, 5 → 1.0
    score = max(0.0, min(1.0, score))

    return JudgeScore(
        score=score,
        reasoning=reasoning,
        correctness=correctness,
        appropriateness=appropriateness,
    )
