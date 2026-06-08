"""Input and output guardrails for the agent endpoint.

These are defense-in-depth. They catch obvious attacks and accidents.
They are NOT a substitute for trained model safety, system prompts,
or audit logging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    reason: str = ""  # human-readable explanation if blocked
    matched_pattern: str = ""  # which pattern fired


# ============================================================
# Input guardrails
# ============================================================


# Known prompt-injection patterns. NOT exhaustive — sophisticated attackers
# will rephrase. This catches lazy attacks and surfaces probing attempts in logs.
_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+(instructions|prompts)", re.IGNORECASE),
    re.compile(r"ignore\s+the\s+above", re.IGNORECASE),
    re.compile(r"disregard\s+(your|the)\s+(instructions|system\s+prompt)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a\s+different\s+(assistant|ai|model)", re.IGNORECASE),
    re.compile(r"reveal\s+(your|the)\s+(system\s+prompt|instructions|hidden)", re.IGNORECASE),
    re.compile(r"new\s+role\s*:\s*", re.IGNORECASE),
]

# Patterns that suggest the user is trying to extract things they shouldn't.
_DATA_EXTRACTION_PATTERNS = [
    re.compile(r"show\s+me\s+(all|every)\s+(user|customer|password|api\s+key)", re.IGNORECASE),
    re.compile(r"dump\s+(the|all)\s+(database|table)", re.IGNORECASE),
    re.compile(r"list\s+all\s+(admin|root|privileged)", re.IGNORECASE),
]

# Maximum acceptable repetition (suggests adversarial padding to overflow context)
_MAX_CHAR_REPETITION = 100


def check_input(prompt: str) -> GuardrailResult:
    """Scan prompt for known attack patterns. Blocks if matches found."""

    # Check 1: prompt injection patterns
    for pattern in _PROMPT_INJECTION_PATTERNS:
        match = pattern.search(prompt)
        if match:
            return GuardrailResult(
                passed=False,
                reason="prompt_injection_pattern_detected",
                matched_pattern=match.group(0)[:100],
            )

    # Check 2: data extraction patterns
    for pattern in _DATA_EXTRACTION_PATTERNS:
        match = pattern.search(prompt)
        if match:
            return GuardrailResult(
                passed=False,
                reason="suspicious_data_extraction_request",
                matched_pattern=match.group(0)[:100],
            )

    # Check 3: character repetition attack
    # Detects strings like "AAAAAAAAA..." used to overflow context or jailbreak
    for char in set(prompt):
        if prompt.count(char) > _MAX_CHAR_REPETITION and char.isalnum():
            return GuardrailResult(
                passed=False,
                reason="excessive_character_repetition",
                matched_pattern=f"char={char!r} count={prompt.count(char)}",
            )

    return GuardrailResult(passed=True)


# ============================================================
# Output guardrails
# ============================================================


# Patterns that suggest the model is about to leak sensitive content.
_PII_PATTERNS = [
    # US SSN format: XXX-XX-XXXX
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Credit card-ish: 16 digits, possibly with spaces or dashes
    re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
    # Common API key prefixes (truncated to first 8 chars for safety in logs)
    re.compile(r"\b(sk-ant-|sk-|AKIA|ghp_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9_\-]{20,}", re.IGNORECASE),
    # Email-like sequences with explicit "leaked from training" patterns
    # (deliberately narrow to avoid false-positive on user-discussed emails)
]

# Phrases that suggest the model is going off-script
_OFF_SCRIPT_PATTERNS = [
    re.compile(r"my\s+(system|hidden)\s+prompt\s+(is|says|tells\s+me)", re.IGNORECASE),
    re.compile(r"i\s+was\s+(instructed|told)\s+(to|not\s+to)\s+reveal", re.IGNORECASE),
    re.compile(r"the\s+secret\s+(key|password|token)\s+is", re.IGNORECASE),
]


def check_output(response_text: str) -> GuardrailResult:
    """Scan agent response for leaks. Blocks if matches found.

    This is the LAST defense before the response reaches the user.
    If a pattern fires here, audit the agent's behavior — it produced
    something we'd rather not have shipped.
    """

    # Check 1: PII patterns
    for pattern in _PII_PATTERNS:
        match = pattern.search(response_text)
        if match:
            return GuardrailResult(
                passed=False,
                reason="pii_leak_detected",
                matched_pattern=match.group(0)[:8] + "[REDACTED]",  # don't log the actual PII
            )

    # Check 2: model going off-script
    for pattern in _OFF_SCRIPT_PATTERNS:
        match = pattern.search(response_text)
        if match:
            return GuardrailResult(
                passed=False,
                reason="model_went_off_script",
                matched_pattern=match.group(0)[:100],
            )

    return GuardrailResult(passed=True)
