"""Structured output — get typed Pydantic objects back from Gemini."""

from __future__ import annotations

from enum import StrEnum

from api.config import get_settings
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ============================================================
# What we want the LLM to produce
# ============================================================


class Category(StrEnum):
    FOOD = "food"
    TRANSPORT = "transport"
    UTILITIES = "utilities"
    ENTERTAINMENT = "entertainment"
    SHOPPING = "shopping"
    SALARY = "salary"
    TRANSFER = "transfer"
    OTHER = "other"


class TransactionCategorization(BaseModel):
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    merchant_name: str | None = Field(
        default=None, description="Cleaned merchant name if identifiable"
    )
    is_recurring: bool = Field(description="Is this likely a recurring payment?")
    reasoning: str = Field(description="Brief explanation of classification")


# ============================================================
# Make the call
# ============================================================


DESCRIPTIONS = [
    "UPI/SWIGGY/swiggy*food/IDFC0040101",
    "UPI/UBER/uber.com/HDFC0000123",
    "NEFT-SALARY-MAY26/INDUS0000001",
    "BILLPAY/AIRTEL/POSTPAID/JUL26",
    "UPI/AMAZON/RAZORPAY/8472HJ",
    "ATM-WDL-AXIS-MUMBAI",
    "UPI-RAHUL-VERMA-EMI-LOAN",
]


def analyze_transaction(
    client: genai.Client, model: str, description: str
) -> TransactionCategorization:
    response = client.models.generate_content(
        model=model,
        contents=f"Analyze this product review:\n\n{description}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TransactionCategorization,
        ),
    )
    # The SDK already validated against the schema; .parsed gives the Pydantic object
    return response.parsed  # type: ignore[return-value]


def main() -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    for i, description in enumerate(DESCRIPTIONS, start=1):
        analysis = analyze_transaction(client, settings.gemini_default_model, description)
        print(f"=== Transaction {i} ===")
        print(f"  text: {description[:60]}...")
        print(f"  category: {analysis.category.value}")
        print(f"  confidence: {analysis.confidence}")
        print(f"  merchant name: {analysis.merchant_name}")
        print(f"  is recurring: {analysis.is_recurring}")
        print(f"  reasoning: {analysis.reasoning}")
        print()


if __name__ == "__main__":
    main()
