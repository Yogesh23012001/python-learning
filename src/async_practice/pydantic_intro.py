"""Hour 3 — Pydantic v2 fundamentals."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class Payment(BaseModel):
    amount: float
    currency: str
    user_id: int
    metadata: dict[str, Any] = {}

    @field_validator("currency")
    @classmethod
    def currency_must_be_uppercase(cls, v: str) -> str:
        """Normalize and validate currency code."""
        v = v.upper()
        if v not in {"INR", "USD", "EUR", "GBP"}:
            raise ValueError(f"unsupported currency: {v}")
        return v  # return the (possibly modified) value

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount must be positive")
        return v

    @model_validator(mode="after")
    def fraud_check(self) -> "Payment":
        """Cross-field validation after all fields are set."""
        if self.currency == "INR" and self.amount > 200000:
            raise ValueError("INR payments above 2 lakh require additional verification")
        return self

    @field_validator("user_id")
    @classmethod
    def user_id_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("user_id must be a positive integer")
        return v

    @model_validator(mode="after")
    def check_currency_amount_combination(self) -> "Payment":
        if (
            self.currency == "INR"
            and self.metadata.get("verified") != "true"
            and self.amount > 50000
        ):
            raise ValueError("INR payments above 50,000 require verified user")
        return self


# 1. Normal payment — succeeds
p1 = Payment(amount=100, currency="usd", user_id=42)
print(f"OK: {p1}")

# 2. INR 100000 without verification — should fail
try:
    Payment(amount=100000, currency="INR", user_id=42)
except Exception as e:
    print(f"\nExpected fraud rule fail:\n{e}")

# 3. INR 100000 WITH verification — should succeed
p3 = Payment(amount=100000, currency="INR", user_id=42, metadata={"verified": "true"})
print(f"\nOK with verification: {p3}")


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = Field(default=1024, ge=1, le=200_000)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages list cannot be empty")
        return v


try:
    ChatRequest(model="claude", messages=[], max_tokens=100)
except Exception as e:
    print(f"Empty messages:\n{e}")
