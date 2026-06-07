# Evaluation Dataset

This directory contains the eval suite for the agent service. The dataset is
the encoded definition of "correct behavior" — what the agent SHOULD do
across a curated set of input prompts.

## Schema (per example)

Each line in `dataset.jsonl` is a JSON object with:

- `id`: stable identifier, never reused even if example is deleted
- `category`: tag from the controlled vocabulary below
- `prompt`: the user input
- `expected_tools`: list of tool names that should be called, OR `[]` for "no
  tools expected", OR `null` for "don't grade tool selection"
- `expected_args`: optional dict of expected arguments per tool, for stricter checking
- `must_contain`: substrings that MUST appear in the final text response
- `must_not_contain`: substrings that MUST NOT appear (red flags)
- `expected_outcome`: `completed | refused | max_iterations | cost_cap | tool_error`
- `notes`: human-readable explanation, not used by graders

## Categories (controlled vocabulary)

| Category | What it tests |
|---|---|
| `tool_selection_single` | One tool, clear ask, expected to fire |
| `tool_selection_none` | No tool relevant, should answer from prior knowledge |
| `multi_tool_composition` | 2+ tools chained, dependency between calls |
| `disambiguation` | Ambiguous reference — should ask or pick conservatively |
| `refusal` | Should decline (out of scope, harmful, impossible) |
| `math` | Should route through `calculate`, not do arithmetic in head |
| `tool_error_handling` | Tool returns error — agent should report gracefully |
| `edge_case` | Hostile input, very short, prompt injection attempts |

## How to extend

When adding a new example:

1. Pick the category. If your example doesn't fit existing categories, propose
   a new one in a separate PR before adding examples to it. New categories
   without examples are noise; categories without examples don't help grading.
2. Use a fresh, never-reused `id`. Format: `<category_short>_<NNN>`.
3. Describe the example in `notes` in one sentence. If you need a paragraph,
   the example is too complex — split it.
4. Run the agent once on the example BEFORE committing. Make sure the
   expected behavior is achievable with the current tool set.

## What this dataset is NOT

- Not exhaustive — 15-20 examples can't cover every input the agent will see.
- Not a substitute for human review — refusal cases especially need eyes.
- Not stable forever — when you add a new tool or change descriptions,
  some examples will need updating.

## Versioning

When you make a backward-incompatible change to schema or category vocabulary,
bump the version comment at the top of `dataset.jsonl` and update graders
to handle both versions during the transition.