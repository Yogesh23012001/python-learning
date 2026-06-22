"""Ragas evaluation harness for the RAG pipeline.

Maps the four metrics to the two failure modes:
  RETRIEVAL: context_precision, context_recall
  GENERATION: faithfulness, answer_relevancy

This is the SETUP — a tiny test set to prove the harness runs. Thursday's
50-question eval reuses this harness across all 5 retrieval modes.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings

# ragas 0.4.3 (latest) hard-imports Vertex AI classes that langchain-community
# 0.4.x removed, and this project is pinned to the langchain 1.x stack — it
# can't downgrade (other deps need tenacity 9.x / langchain-text-splitters 1.x).
# We use the Anthropic judge, not Vertex, so stub the dead modules before
# importing ragas. MUST run before `import ragas`.
for _name, _attr in (
    ("langchain_community.chat_models.vertexai", "ChatVertexAI"),
    ("langchain_community.llms.vertexai", "VertexAI"),
):
    try:
        __import__(_name)
    except ModuleNotFoundError:
        _stub = types.ModuleType(_name)
        setattr(_stub, _attr, type(_attr, (), {}))
        sys.modules[_name] = _stub

from ragas import evaluate  # noqa: E402  (must follow the Vertex stub above)
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from rag.generator import answer_question  # noqa: E402

# Load the project .env (one dir up from rag/) so ANTHROPIC_API_KEY is available
# when this module runs standalone — the same key source the evals use.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit("ANTHROPIC_API_KEY not set in .env")

# Ragas needs a judge LLM and an embedder. Point both at what we already use.
JUDGE_LLM = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    temperature=0,
)
EVAL_EMBEDDINGS = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def build_eval_dataset(qa_pairs: list[dict], *, mode: str = "dense") -> Dataset:
    """Run the RAG pipeline on each question, assemble the Ragas dataset.

    Each qa_pair: {"question": str, "ground_truth": str}
    Ragas needs: question, answer, contexts (retrieved), ground_truth.
    """
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for pair in qa_pairs:
        result = answer_question(pair["question"], mode=mode)
        rows["question"].append(pair["question"])
        rows["answer"].append(result.answer)
        rows["contexts"].append([c.text for c in result.retrieved])
        rows["ground_truth"].append(pair["ground_truth"])
    return Dataset.from_dict(rows)


def run_eval(qa_pairs: list[dict], *, mode: str = "dense"):
    """Run Ragas on the pipeline for a given retrieval mode."""
    dataset = build_eval_dataset(qa_pairs, mode=mode)
    result = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=JUDGE_LLM,
        embeddings=EVAL_EMBEDDINGS,
    )
    return result


# A TINY test set — just to prove the harness runs. Thursday: 50 questions.
TINY_QA = [
    {
        "question": "How do I use XML tags in prompts?",
        "ground_truth": "Wrap different content types in XML tags like <instructions>, <context>, and <example> to help Claude parse complex prompts unambiguously. Use consistent, descriptive tag names and nest them when content has a natural hierarchy.",
    },
    {
        "question": "How does prompt caching reduce costs?",
        "ground_truth": "Prompt caching lets you reuse cached prompt prefixes. Cache reads cost significantly less than uncached input tokens, so reusing stable content like system instructions or large context reduces cost.",
    },
    {
        "question": "How does Claude decide whether to call a tool?",
        "ground_truth": "With tool_choice set to auto, Claude decides each turn whether to call a tool based on whether the request maps to a tool's capability and the answer isn't already in context. It responds directly for stable knowledge and conversational turns.",
    },
]


if __name__ == "__main__":
    print("Running tiny Ragas eval (dense mode)...\n")
    result = run_eval(TINY_QA, mode="dense")
    print("\n=== RESULTS (dense) ===")
    print(result)
