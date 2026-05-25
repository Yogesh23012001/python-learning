"""Shared Prometheus metrics for the llm package.

Defined in one module so the default REGISTRY only sees a single Counter
named `llm_calls_total`. Both the low-level provider client and the
higher-level router import the same instance from here.
"""

from __future__ import annotations

from prometheus_client import Counter

# One increment per provider-call attempt (so retries are visible).
# Labels:
#   provider: "gemini" | "anthropic" | ... — which SDK was used
#   model:    e.g. "gemini-2.5-flash"
#   outcome:  "success" | "failed"
llm_calls_total = Counter(
    "llm_calls_total",
    "LLM completion calls",
    labelnames=("provider", "model", "outcome"),
)
