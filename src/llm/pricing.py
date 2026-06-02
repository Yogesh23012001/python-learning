"""LLM pricing tables and cost calculation.

Prices are in USD per million tokens. Update periodically against
provider documentation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float


PRICING: dict[str, ModelPricing] = {
    # Gemini
    "gemini-2.5-flash": ModelPricing(0.30, 2.50),
    "gemini-2.5-flash-lite": ModelPricing(0.30, 2.50),
    "gemini-2.5-pro": ModelPricing(1.25, 10.00),
    "gemini-2.0-flash": ModelPricing(0.10, 0.40),
    # Claude (for future use)
    "claude-haiku-4-5": ModelPricing(1.00, 5.00),
    "claude-sonnet-4-6": ModelPricing(3.00, 15.00),
    # OpenAI (for future use)
    "gpt-4.1-mini": ModelPricing(0.40, 1.60),
}


def calculate_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Return USD cost for a single call.

    Raises:
        KeyError: If model is not in the pricing table.
    """
    pricing = PRICING[model]
    return (
        input_tokens * pricing.input_per_million / 1_000_000
        + output_tokens * pricing.output_per_million / 1_000_000
    )
