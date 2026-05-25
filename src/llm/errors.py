"""LLM-specific error types."""

from __future__ import annotations


class LLMError(Exception):
    """Base for all LLM call errors."""


class LLMRetryableError(LLMError):
    """The provider is having transient trouble; safe to retry."""


class LLMRateLimitError(LLMRetryableError):
    """Too many requests."""


class LLMOverloadedError(LLMRetryableError):
    """Provider is overloaded; backoff and retry."""


class LLMPermanentError(LLMError):
    """Don't retry — input or account state is wrong."""


class LLMQuotaExceededError(LLMPermanentError):
    """API tier quota exhausted (free tier daily cap, paid tier monthly cap)."""


class LLMContextTooLongError(LLMPermanentError):
    """Input exceeded the model's context window."""


class LLMContentBlockedError(LLMPermanentError):
    """Safety filter blocked the request or response."""


class LLMEmptyResponseError(LLMPermanentError):
    """Model returned no usable text (e.g., safety refusal)."""
