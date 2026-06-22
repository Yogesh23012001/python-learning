Retrieval quality (did we fetch the right context?)
1. Context Precision — Of the chunks we retrieved, how many are actually relevant? High precision = little noise in the retrieved set. This is the metric that caught HyDE's real effect — remember "1 relevant of 8" (dense) vs "7 relevant of 8" (HyDE)? That ratio is context precision. Low precision means you're feeding the LLM junk alongside the signal.
2. Context Recall — Of all the information needed to answer, how much did we actually retrieve? High recall = we didn't miss relevant chunks. Precision and recall trade off: retrieve 20 chunks → high recall (you got everything) but low precision (lots of noise); retrieve 2 → high precision (both relevant) but maybe low recall (missed something). This is the classic precision/recall tension, now in retrieval.
Generation quality (did the LLM answer faithfully from the context?)
3. Faithfulness — Is every claim in the answer supported by the retrieved context? This is the hallucination detector. Remember the caching answer where you weren't sure if "cache reads cost less" came from a chunk or from training memory? Faithfulness automates exactly that check — it decomposes the answer into claims and verifies each against the context. Low faithfulness = the model is adding unsupported facts.
4. Answer Relevance — Does the answer actually address the question asked? A faithful answer can still be irrelevant (grounded in context but not answering the question). This catches "technically true, didn't answer."




# Ragas Evaluation

## The four metrics = my two failure modes

The 2×2 that organizes everything — two failure modes, two metrics each:

|                | "Did we fetch the right context?" | "Did we answer faithfully from it?" |
|----------------|-----------------------------------|-------------------------------------|
| **stage**      | RETRIEVAL                         | GENERATION                          |
| **metrics**    | Context Precision, Context Recall | Faithfulness, Answer Relevance      |

The whole point: these **automate the eyeball checks I've been doing by hand all
week.** Two concrete ones from this week:
- **Context Precision = HyDE's "1 of 8" vs "7 of 8".** When I compared dense vs
  HyDE on the "Claude keeps forgetting" query, I counted relevant chunks in the
  top-k by eye. That ratio *is* context precision. Ragas computes it for me.
- **Faithfulness = the caching answer I wasn't sure about.** When the generator
  said "cache reads cost less" I couldn't tell if that came from a retrieved
  chunk or the model's training memory. Faithfulness is exactly that check,
  automated — decompose the answer into claims, verify each against the context.

## What each measures (my words)

- **Context Precision** — relevant of retrieved. Noise check: is the top-k mostly
  signal or padded with strays?
- **Context Recall** — retrieved of needed. Completeness check: did we miss
  relevant chunks? Needs ground truth (you can't measure "of all needed" without
  knowing what "all" is).
- **Faithfulness** — claims supported by context. Hallucination detector: is every
  statement in the answer grounded in what we retrieved?
- **Answer Relevance** — answer addresses the question. On-topic check: catches
  the "faithful but didn't actually answer" case.

## How Ragas works

LLM-as-judge — the **same eval machinery as Week 2**, pointed at RAG. It doesn't
string-match; it asks a judge model to reason:
- *Faithfulness*: decompose the answer into atomic claims, then check each claim
  against the retrieved context. Score = supported claims / total claims.
- *Answer Relevance*: generate questions *from* the answer, embed them, and
  compare to the original question — a relevant answer implies the original
  question. (This is why an embedder is needed, not just a judge LLM.)
- *Context Precision/Recall*: judge each retrieved chunk against the ground-truth
  answer for relevance/coverage.

Each row needs four fields: **question, answer, contexts (retrieved), ground_truth.**
Judge = **Haiku** (`claude-haiku-4-5-20251001`), embedder = **bge-small** — the
same models already in the pipeline, so the eval costs Haiku calls, not Opus.

## Setup status

`rag/eval_harness.py` built and importing. `build_eval_dataset` runs the RAG
pipeline per question (any of the 5 modes) and assembles the Ragas dataset;
`run_eval` scores the 4 metrics. A `TINY_QA` (3 questions) proves the harness
loads. **Thursday: a 50-question set, run across all 5 retrieval modes
(dense / hybrid / rerank / hyde / contextual), compared head-to-head.**

Caveats to carry into Thursday's numbers:
- **Contextual mode is incomplete: 586/677 chunks have a `contextual_embedding`**
  (the rest failed the cross-run text-match and are `NULL`). Contextual retrieval
  only sees the 586, so its numbers carry that gap — note it, don't compare as if
  it were a full re-index.
- **Ragas vs the toolchain:** ragas 0.4.3 (latest) doesn't import under this
  project's langchain 1.x stack — it hard-imports Vertex AI modules that
  `langchain-community 0.4.x` removed. Fix in `eval_harness.py`: stub the dead
  Vertex modules in `sys.modules` *before* `import ragas` (I don't use Vertex —
  judge is Anthropic). Import is confirmed; the actual `evaluate()` run under
  langchain 1.x is **not yet verified** — if it breaks at runtime, the fallback
  is to compute the 4 metrics with direct judge calls and drop ragas entirely.
- **Prereqs:** the `rag-postgres-1` pgvector container must be up, and
  `ANTHROPIC_API_KEY` loads from `.env` (added `load_dotenv` to the harness).