"""Sanity-check the eval dataset.

Run from project root:
    uv run python evals/validate_dataset.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DATASET = Path(__file__).parent / "datasets.jsonl"

REQUIRED_FIELDS = {
    "id",
    "category",
    "prompt",
    "expected_tools",
    "must_contain",
    "must_not_contain",
    "expected_outcome",
}


def main() -> None:
    with DATASET.open() as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(examples)} examples from {DATASET.name}")

    # Category distribution
    categories = Counter(ex["category"] for ex in examples)
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # ID uniqueness
    ids = [ex["id"] for ex in examples]
    duplicates = [i for i, c in Counter(ids).items() if c > 1]
    if duplicates:
        print(f"\nDUPLICATE IDS: {duplicates}")
    else:
        print(f"\nAll {len(ids)} IDs unique")

    # Required fields
    missing_report = []
    for ex in examples:
        missing = REQUIRED_FIELDS - ex.keys()
        if missing:
            missing_report.append((ex.get("id", "<no id>"), missing))
    if missing_report:
        print(f"\nMissing fields in {len(missing_report)} examples:")
        for ex_id, missing in missing_report:
            print(f"  {ex_id}: missing {sorted(missing)}")
    else:
        print("\nAll examples have required fields")


if __name__ == "__main__":
    main()
