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



  -Air myproject % uv run python -m rag.eval_async

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 7218.55it/s]

Evaluated 28 questions in 221.4s (dense mode)

Q: How do I structure an effective prompt for Claude?
   faith=1.0  prec=1.0  recall=0.5  rel=1.0

Q: How do I use XML tags to organize a prompt?
   faith=1.0  prec=1.0  recall=0.857  rel=1.0

Q: How do I define a custom tool for Claude to call?
   faith=1.0  prec=0.2  recall=0.5  rel=0.0

Q: How do I cache a large system prompt to reduce cos
   faith=0.75  prec=1.0  recall=0.667  rel=1.0

Q: How do I keep a long conversation from exceeding t
   faith=1.0  prec=0.8  recall=0.833  rel=1.0

Q: How do I get Claude to cite sources from a documen
   faith=1.0  prec=1.0  recall=1.0  rel=1.0

Q: How do I include an image in a message to Claude?
   faith=1.0  prec=0.8  recall=1.0  rel=1.0

Q: How do I force Claude's response into a specific J
   faith=1.0  prec=0.6  recall=0.5  rel=1.0

Q: How do I stream a long response token by token?
   faith=1.0  prec=0.4  recall=0.333  rel=1.0

Q: How do I process many requests asynchronously and 
   faith=0.933  prec=0.8  recall=0.625  rel=1.0

Q: How do I have Claude read a PDF?
   faith=1.0  prec=1.0  recall=1.0  rel=1.0

Q: How do I write good test cases for evaluating Clau
   faith=0.857  prec=0.2  recall=0.455  rel=1.0

Q: What are text embeddings and when would I use them
   faith=1.0  prec=0.4  recall=0.538  rel=1.0

Q: What is the difference between extended thinking a
   faith=0.944  prec=0.8  recall=0.714  rel=1.0

Q: Why would I count tokens before sending a request,
   faith=1.0  prec=0.8  recall=0.5  rel=1.0

Q: How do prompt caching and batch processing each re
   faith=0.833  prec=0.6  recall=0.889  rel=1.0

Q: When should I use tool use versus structured outpu
   faith=1.0  prec=0.4  recall=0.667  rel=0.0

Q: What does `cache_control` with `"type": "ephemeral
   faith=0.909  prec=0.6  recall=0.833  rel=1.0

Q: What are the possible values of the `tool_choice` 
   faith=0.8  prec=0.4  recall=0.833  rel=1.0

Q: Is the `budget_tokens` parameter still supported?
   faith=0.875  prec=0.8  recall=0.5  rel=1.0

Q: What cost discount does the Message Batches API gi
   faith=1.0  prec=0.4  recall=1.0  rel=1.0

Q: What does setting `strict: true` on a tool do?
   faith=1.0  prec=0.6  recall=1.0  rel=1.0

Q: Why does Claude keep forgetting what we talked abo
   faith=1.0  prec=0.0  recall=0.167  rel=0.0

Q: My API bill is huge — how do I make it cheaper?
   faith=1.0  prec=1.0  recall=0.625  rel=1.0

Q: How do I stop Claude from making up facts that are
   faith=1.0  prec=0.0  recall=0.4  rel=0.0

Q: Can Claude look at a screenshot and tell me what's
   faith=1.0  prec=0.8  recall=0.7  rel=1.0

Q: What is the exact price per million tokens for Cla
   faith=1.0  prec=0.4  recall=1.0  rel=0.0

Q: How do I fine-tune Claude on my own training data?
   faith=1.0  prec=0.0  recall=1.0  rel=0.0

=== AVERAGES (dense) ===
  faithfulness: 0.961
  context_precision: 0.600
  context_recall: 0.701
  answer_relevance: 0.786