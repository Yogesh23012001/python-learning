"""LLM-based developer summary generation."""

from __future__ import annotations

import structlog
from github_fetcher.models import UserScore

from llm.router import LLMRouter

logger = structlog.get_logger(__name__)


SUMMARY_PROMPT_TEMPLATE = """You are writing a one-paragraph developer summary for a hiring dashboard.

Given the GitHub profile data below, produce a single concise paragraph (2-4 sentences) describing:
- Their overall activity level (use the score and repo count as signals)
- Their reach (followers, stars on their repos)
- Whether they appear to be an active open-source contributor vs a casual user

Write in third person. Be specific. Don't invent details not present in the data.
Don't use markdown formatting. Don't add a header. Just the paragraph.

Profile:
- Login: {login}
- Display name: {name}
- Public repos: {public_repos}
- Total stars across repos: {total_stars}
- Total forks across repos: {total_forks}
- Followers: {followers}
- Computed score: {score}
"""


async def generate_developer_summary(
    router: LLMRouter,
    *,
    user_score: UserScore,
    max_output_tokens: int = 500,
) -> str:
    """Use the LLM to produce a developer-summary paragraph."""
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        login=user_score.login,
        name=user_score.name or "(no name set)",
        public_repos=user_score.public_repos,
        total_stars=user_score.total_stars,
        total_forks=user_score.total_forks,
        followers=user_score.followers,
        score=user_score.score,
    )
    response = await router.generate(prompt, max_output_tokens=max_output_tokens)
    logger.info(
        "developer_summary_generated",
        login=user_score.login,
        summary_length=len(response.text),
        cost_usd=response.cost_usd,
    )
    return response.text.strip()
