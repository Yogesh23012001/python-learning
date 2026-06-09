"""Summary-specific grader.

Scores along four dimensions:
  - specificity: does the summary mention named entities and real numbers?
  - factual_grounding: do the factual claims match real data?
  - hallucination_risk: are there unverifiable claims?
  - readability: is the prose coherent?

Specificity is computed deterministically. The other three use LLM-as-judge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic


@dataclass(frozen=True)
class SummaryScore:
    """Result of summary grading along four dimensions."""

    specificity: float  # 0.0 - 1.0, deterministic
    factual_grounding: int  # 1-5, from judge
    hallucination_risk: int  # 1-5, from judge (5 = no hallucinated claims, 1 = lots)
    readability: int  # 1-5, from judge

    overall: float  # 0.0 - 1.0 weighted aggregate
    judge_reasoning: str
    error: str | None = None


# ============================================================
# Specificity (deterministic)
# ============================================================


# Patterns for "named entities" in a summary
_REPO_NAME_PATTERN = re.compile(
    r"\b[a-z][a-z0-9_\-]{2,40}\b"
)  # lowercase names like 'nanoGPT', 'llm.c'
_NUMBER_PATTERN = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")  # 1,234 or 1234.5
_LANGUAGE_PATTERN = re.compile(
    r"\b(Python|JavaScript|TypeScript|Go|Rust|C\+\+|C#|Java|Ruby|PHP|Swift|Kotlin|CUDA|C)\b"
)


# Common English words to exclude from the "named entity" count
_STOPWORDS = {
    "with",
    "the",
    "and",
    "for",
    "from",
    "into",
    "their",
    "his",
    "her",
    "she",
    "they",
    "this",
    "that",
    "across",
    "developer",
    "github",
    "user",
    "summary",
    "profile",
    "overview",
    "is",
    "has",
    "are",
    "in",
    "on",
    "at",
    "of",
    "to",
    "by",
    "as",
    "an",
    "a",
    "or",
    "be",
    "but",
    "if",
    "so",
}


def compute_specificity(summary_text: str) -> float:
    """Estimate specificity from named entities + numbers + languages.

    Returns 0.0-1.0. Tuned so:
    - Generic "they have substantial impact" → ~0.1
    - "nanoGPT (59,359 stars) in Python" → ~0.7+
    """
    text_lower = summary_text.lower()

    # Count distinct numeric values (real data, not made-up)
    numbers = set(_NUMBER_PATTERN.findall(summary_text))
    num_distinct_numbers = len(numbers)

    # Count language mentions
    languages = set(_LANGUAGE_PATTERN.findall(summary_text))
    num_languages = len(languages)

    # Count plausible repo-name-like tokens (lowercase, with hyphens/underscores)
    repo_candidates = _REPO_NAME_PATTERN.findall(text_lower)
    distinct_repo_candidates = set(repo_candidates) - _STOPWORDS
    num_repo_names = sum(1 for w in distinct_repo_candidates if "-" in w or "_" in w or len(w) > 6)

    # Composite score
    # Tuning: 2 numbers + 1 language + 2 repo names = "specific" ≈ 0.7
    score = (
        min(num_distinct_numbers, 5) * 0.10
        + min(num_languages, 3) * 0.10
        + min(num_repo_names, 5) * 0.10
    )

    return min(score, 1.0)


# ============================================================
# Judge-based grading
# ============================================================


SUMMARY_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of GitHub user developer summaries. Grade the summary along three dimensions:

1. FACTUAL_GROUNDING (1-5): How well do the factual claims (stars, repos, projects, follower counts) match the source data?
   - 5: All factual claims accurate and grounded
   - 3: Some claims unverifiable but not clearly wrong
   - 1: Multiple clearly wrong or fabricated claims

2. HALLUCINATION_RISK (1-5): How much does the summary make unverifiable interpretive claims (characterizations of the user's "philosophy", "approach", or unstated motivations)?
   - 5: All claims grounded in observable data
   - 3: Some characterization but reasonable inference
   - 1: Significant unsupported characterization or invented details

3. READABILITY (1-5): How well-written is the summary?
   - 5: Clear, well-structured, professional
   - 3: Adequate but could be clearer
   - 1: Confused or hard to read

Respond with ONLY a JSON object:
{
  "factual_grounding": <int>,
  "hallucination_risk": <int>,
  "readability": <int>,
  "reasoning": "<2-3 sentence explanation>"
}

No markdown, no preamble. Just the JSON object."""


async def grade_summary(
    *,
    client: AsyncAnthropic,
    login: str,
    summary_text: str,
    actual_profile_data: dict[str, Any],
    judge_model: str = "claude-sonnet-4-5",
) -> SummaryScore:
    """Score a summary on all four dimensions."""

    # Deterministic: specificity
    specificity = compute_specificity(summary_text)

    # Judge-based: factual grounding, hallucination risk, readability
    judge_prompt = f"""GITHUB USER: {login}

ACTUAL PROFILE DATA (ground truth):
{json.dumps(actual_profile_data, indent=2)}

SUMMARY TO GRADE:
{summary_text}

Grade this summary. Output ONLY the JSON object."""

    try:
        response = await client.messages.create(
            model=judge_model,
            max_tokens=500,
            system=SUMMARY_JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": judge_prompt}],
        )
    except Exception as exc:
        return SummaryScore(
            specificity=specificity,
            factual_grounding=0,
            hallucination_risk=0,
            readability=0,
            overall=0.0,
            judge_reasoning="judge call failed",
            error=str(exc)[:200],
        )

    text = "".join(b.text for b in response.content if b.type == "text").strip()

    try:
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        parsed = json.loads(text)
    except (json.JSONDecodeError, IndexError) as exc:
        return SummaryScore(
            specificity=specificity,
            factual_grounding=0,
            hallucination_risk=0,
            readability=0,
            overall=0.0,
            judge_reasoning=f"failed to parse: {text[:200]}",
            error=str(exc)[:200],
        )

    factual = int(parsed.get("factual_grounding", 0))
    hallucination = int(parsed.get("hallucination_risk", 0))
    readability = int(parsed.get("readability", 0))
    reasoning = parsed.get("reasoning", "")

    # Weighted overall: factual grounding matters most, specificity + hallucination
    # both important. Readability is a tiebreaker.
    overall = (
        0.30 * specificity
        + 0.30 * ((factual - 1.0) / 4.0)
        + 0.25 * ((hallucination - 1.0) / 4.0)
        + 0.15 * ((readability - 1.0) / 4.0)
    )
    overall = max(0.0, min(1.0, overall))

    return SummaryScore(
        specificity=specificity,
        factual_grounding=factual,
        hallucination_risk=hallucination,
        readability=readability,
        overall=overall,
        judge_reasoning=reasoning,
    )
