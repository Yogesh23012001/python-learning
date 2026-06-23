"""Spot-check: do HyDE / contextual rescue the mismatch questions that dense
failed (precision 0.0)?

Hypothesis: the mismatch questions failed because question-space != answer-space.
HyDE (embed a hypothetical answer) should bridge that. The cannot-answer question
is a CONTROL — it should stay failed (the answer genuinely isn't in the corpus).
"""

from __future__ import annotations

import asyncio

from rag.eval_async import evaluate_one

# The questions that scored precision 0.0 under dense
SPOT_QUESTIONS = [
    {
        "question": "Why does Claude keep forgetting what we talked about earlier in a long chat?",
        "ground_truth": "Because the conversation can exceed the context window (token limit); once exceeded, earlier content is dropped or compacted and no longer visible to the model.",
        "category": "mismatch",
    },
    {
        "question": "How do I stop Claude from making up facts that aren't true?",
        "ground_truth": "Ground Claude in provided context and instruct it to answer only from that context and say when it doesn't know; techniques in the reduce-hallucinations guidance (e.g. allow 'I don't know', ask for quotes/citations).",
        "category": "mismatch",
    },
    {
        "question": "How do I fine-tune Claude on my own training data?",
        "ground_truth": "NOT_IN_CORPUS — the docs don't cover fine-tuning; a faithful system should decline.",
        "category": "cannot-answer (CONTROL)",
    },
]

MODES = ["dense", "hyde", "contextual"]


async def _main() -> None:
    for qa in SPOT_QUESTIONS:
        print(f"\n{'='*70}")
        print(f"Q: {qa['question']}")
        print(f"   category: {qa['category']}")
        print(f"{'-'*70}")
        for mode in MODES:
            r = await evaluate_one(qa, mode=mode)
            print(
                f"  {mode:11s}  prec={r['context_precision']:.2f}  "
                f"recall={r['context_recall']:.2f}  "
                f"faith={r['faithfulness']:.2f}  rel={r['answer_relevance']:.1f}"
            )


if __name__ == "__main__":
    asyncio.run(_main())
