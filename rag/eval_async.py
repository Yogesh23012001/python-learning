"""Hand-rolled async RAG eval — the 4 metrics via concurrent, trivially-schema'd
judge calls.

Replaces ragas (which, with Haiku + instructor, ballooned to ~8 min/question via
structured-output retry storms). Here every judge call returns the SMALLEST
possible JSON, so the judge succeeds first try — no retries — and all calls run
concurrently under a semaphore.

Metrics (= the two failure modes):
  RETRIEVAL:  context_precision, context_recall
  GENERATION: faithfulness, answer_relevance
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from rag.generator import answer_question

# Load the project .env (one dir up from rag/) so ANTHROPIC_API_KEY is available
# when this module runs standalone — the same key source the evals use.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit("ANTHROPIC_API_KEY not set in .env")

# The SDK reads ANTHROPIC_API_KEY from the environment automatically.
_client = AsyncAnthropic()
JUDGE = "claude-haiku-4-5-20251001"

# Bound concurrency so we don't hit rate limits. Tune if you see 429s.
_sem = asyncio.Semaphore(8)


async def _judge_bool(prompt: str, key: str) -> bool:
    """One judge call returning a single boolean under `key`. Trivial schema."""
    async with _sem:
        resp = await _client.messages.create(
            model=JUDGE,
            max_tokens=10,  # tiny — just the boolean
            messages=[{"role": "user", "content": prompt}],
        )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return bool(json.loads(text).get(key, False))
    except (json.JSONDecodeError, AttributeError):
        # Fallback: look for "true" in the raw text
        return "true" in text.lower()


async def _judge_claims(answer: str) -> list[str]:
    """Decompose an answer into atomic claims (one judge call)."""
    async with _sem:
        resp = await _client.messages.create(
            model=JUDGE,
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": f"""Break this answer into a list of atomic factual claims (short standalone statements). If the answer declines to answer or says it lacks information, return an empty list.

ANSWER: {answer}

Respond ONLY with JSON: {{"claims": ["claim 1", "claim 2"]}}""",
                }
            ],
        )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text).get("claims", [])
    except (json.JSONDecodeError, AttributeError):
        return []


async def faithfulness(answer: str, contexts: list[str]) -> float:
    """Supported claims / total claims. Each claim checked concurrently."""
    claims = await _judge_claims(answer)
    if not claims:
        # No claims = the model declined. That's faithful (it didn't invent).
        return 1.0
    context_block = "\n\n".join(contexts)
    tasks = [
        _judge_bool(
            f"""CONTEXT:\n{context_block}\n\nCLAIM: {claim}\n\nIs this claim directly supported by the context? Respond ONLY with JSON: {{"supported": true}} or {{"supported": false}}""",
            "supported",
        )
        for claim in claims
    ]
    results = await asyncio.gather(*tasks)
    return sum(results) / len(results)


async def context_precision(question: str, contexts: list[str]) -> float:
    """Fraction of retrieved chunks relevant to the question. Concurrent."""
    if not contexts:
        return 0.0
    tasks = [
        _judge_bool(
            f"""QUESTION: {question}\n\nCHUNK: {ctx}\n\nIs this chunk relevant for answering the question? Respond ONLY with JSON: {{"relevant": true}} or {{"relevant": false}}""",
            "relevant",
        )
        for ctx in contexts
    ]
    results = await asyncio.gather(*tasks)
    return sum(results) / len(results)


async def context_recall(ground_truth: str, contexts: list[str]) -> float:
    """Of the claims in the GROUND TRUTH, how many are supported by retrieved context?

    This is recall: did we retrieve enough to cover the correct answer?
    """
    if ground_truth.startswith("NOT_IN_CORPUS"):
        return 1.0  # nothing to recall; not applicable
    gt_claims = await _judge_claims(ground_truth)
    if not gt_claims:
        return 1.0
    context_block = "\n\n".join(contexts)
    tasks = [
        _judge_bool(
            f"""CONTEXT:\n{context_block}\n\nCLAIM: {claim}\n\nIs this claim supported by the context? Respond ONLY with JSON: {{"supported": true}} or {{"supported": false}}""",
            "supported",
        )
        for claim in gt_claims
    ]
    results = await asyncio.gather(*tasks)
    return sum(results) / len(results)


async def answer_relevance(question: str, answer: str) -> bool:
    """Does the answer address the question? (Single judgment, 0/1.)"""
    return await _judge_bool(
        f"""QUESTION: {question}\n\nANSWER: {answer}\n\nDoes the answer directly address the question (not decline, not go off-topic)? Respond ONLY with JSON: {{"relevant": true}} or {{"relevant": false}}""",
        "relevant",
    )


async def evaluate_one(qa: dict, *, mode: str) -> dict:
    """Run the pipeline once, score all 4 metrics concurrently."""
    result = answer_question(qa["question"], mode=mode)  # pipeline call (sync)
    contexts = [c.text for c in result.retrieved]

    faith, prec, recall, rel = await asyncio.gather(
        faithfulness(result.answer, contexts),
        context_precision(qa["question"], contexts),
        context_recall(qa["ground_truth"], contexts),
        answer_relevance(qa["question"], result.answer),
    )
    return {
        "question": qa["question"],
        "category": qa.get("category", "?"),
        "faithfulness": round(faith, 3),
        "context_precision": round(prec, 3),
        "context_recall": round(recall, 3),
        "answer_relevance": float(rel),
    }


async def evaluate_set(qa_pairs: list[dict], *, mode: str) -> list[dict]:
    """Evaluate a whole set for one mode. Questions run concurrently too."""
    tasks = [evaluate_one(qa, mode=mode) for qa in qa_pairs]
    return await asyncio.gather(*tasks)


# add to rag/eval_async.py


async def _main() -> None:
    import time

    from rag.eval_questions import EVAL_QA

    t0 = time.perf_counter()
    results = await evaluate_set(EVAL_QA, mode="dense")
    elapsed = time.perf_counter() - t0

    print(f"\nEvaluated {len(results)} questions in {elapsed:.1f}s (dense mode)\n")
    for r in results:
        print(f"Q: {r['question'][:50]}")
        print(
            f"   faith={r['faithfulness']}  prec={r['context_precision']}  "
            f"recall={r['context_recall']}  rel={r['answer_relevance']}\n"
        )

    # Averages
    n = len(results)
    print("=== AVERAGES (dense) ===")
    for m in ["faithfulness", "context_precision", "context_recall", "answer_relevance"]:
        avg = sum(r[m] for r in results) / n
        print(f"  {m}: {avg:.3f}")


if __name__ == "__main__":
    asyncio.run(_main())
