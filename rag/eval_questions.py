"""28-question evaluation set for the RAG pipeline.

Spread across all 15 corpus docs and tagged by category — the categories are the
point, because each stresses a different retrieval mode:

  factual    (12) — bread-and-butter how-to; baseline, every mode should handle
  conceptual  (5) — synthesis across chunks; stresses recall + faithfulness
  exact_term  (5) — API params / acronyms; hybrid (BM25) territory
  casual      (4) — phrased unlike the docs; HyDE / contextual territory
  cannot      (2) — not in the corpus; faithfulness / decline test
                                                                 -> 28 total

Each row: {question, ground_truth, category, doc}.
  - The harness only reads `question` + `ground_truth`.
  - `category` is metadata for the per-category breakdown.
  - `doc` is the doc_id the answer should come from (None for cannot-answer) —
    handy for checking whether retrieval fetched the right source.

NOTE: the ground_truths are reference answers written from the docs' topics, not
verbatim chunk text. Spot-check a few against the live docs before trusting the
absolute metric numbers; they're solid for *relative* mode comparison either way.
"""

from __future__ import annotations

EVAL_QA: list[dict] = [
    # ============================================================
    # FACTUAL / HOW-TO (12) — one per doc, all modes should do okay
    # ============================================================
    {
        "question": "How do I structure an effective prompt for Claude?",
        "ground_truth": "Put long context/data near the top of the prompt and your query and instructions at the end; be clear and direct about the task; separate distinct parts (instructions, context, examples) with XML tags or headings; and give Claude a role via the system prompt. Clear structure improves accuracy and consistency.",
        "category": "factual",
        "doc": "prompt-engineering/overview",
    },
    {
        "question": "How do I use XML tags to organize a prompt?",
        "ground_truth": "Wrap distinct content types in named XML tags (e.g. <instructions>, <context>, <example>) so Claude can tell them apart unambiguously. Use consistent, descriptive tag names, nest them when content has a natural hierarchy, and refer back to tagged sections by name in your instructions.",
        "category": "factual",
        "doc": "prompt-engineering/claude-prompting-best-practices",
    },
    {
        "question": "How do I define a custom tool for Claude to call?",
        "ground_truth": "Provide a tool definition with a name, a clear description of when to use it, and an input_schema (JSON Schema for the parameters). Pass it in the tools array; Claude may return a tool_use block, which you execute and send back as a tool_result. Descriptions that state when to call the tool improve selection.",
        "category": "factual",
        "doc": "tool-use/overview",
    },
    {
        "question": "How do I cache a large system prompt to reduce cost?",
        "ground_truth": 'Add cache_control {"type": "ephemeral"} to the block you want cached (e.g. the system prompt), or use top-level cache_control to auto-cache the last cacheable block. The first request writes the cache (~1.25x cost); later requests within the TTL read it at ~10% of input cost. Keep the cached prefix byte-identical across requests.',
        "category": "factual",
        "doc": "build-with-claude/prompt-caching",
    },
    {
        "question": "How do I keep a long conversation from exceeding the context window?",
        "ground_truth": "Track token usage against the model's context window; as you approach the limit, summarize or compact earlier turns (server-side compaction is available on supported models), trim stale content, or split the conversation. The context window is the maximum number of input plus output tokens the model can consider at once.",
        "category": "factual",
        "doc": "build-with-claude/context-windows",
    },
    {
        "question": "How do I get Claude to cite sources from a document?",
        "ground_truth": "Provide the source as a document content block with citations enabled (citations: {enabled: true}). Claude then returns cited claims that point to specific locations in the document, so you can verify which part each statement came from.",
        "category": "factual",
        "doc": "build-with-claude/citations",
    },
    {
        "question": "How do I include an image in a message to Claude?",
        "ground_truth": "Send an image content block in the user message, with source as base64 (type, media_type, data) or a URL, optionally alongside a text block asking about it. Supported formats include JPEG, PNG, GIF, and WebP.",
        "category": "factual",
        "doc": "build-with-claude/vision",
    },
    {
        "question": "How do I force Claude's response into a specific JSON schema?",
        "ground_truth": "Use output_config.format with a json_schema describing the shape you want (or messages.parse() with a Pydantic/Zod model). The response is constrained to valid JSON matching the schema. For tool arguments specifically, set strict: true on the tool.",
        "category": "factual",
        "doc": "build-with-claude/structured-outputs",
    },
    {
        "question": "How do I stream a long response token by token?",
        "ground_truth": "Use the streaming API (the messages.stream() helper, or stream: true). The server emits Server-Sent Events — message_start, content_block_delta (text/thinking deltas), message_delta, message_stop. Accumulate deltas as they arrive, or call get_final_message()/finalMessage() for the complete response. Streaming also avoids timeouts on large max_tokens.",
        "category": "factual",
        "doc": "build-with-claude/streaming",
    },
    {
        "question": "How do I process many requests asynchronously and at lower cost?",
        "ground_truth": "Use the Message Batches API: submit many requests (each with a custom_id and params) in one batch, poll until processing_status is 'ended', then retrieve results by custom_id. Batches run asynchronously (most finish within an hour, max 24h) at a 50% discount on token cost.",
        "category": "factual",
        "doc": "build-with-claude/batch-processing",
    },
    {
        "question": "How do I have Claude read a PDF?",
        "ground_truth": "Send the PDF as a document content block — base64 data with media_type application/pdf, a URL, or a Files API file_id — alongside a text prompt. Claude can read both the text and the visual layout of the PDF pages.",
        "category": "factual",
        "doc": "build-with-claude/pdf-support",
    },
    {
        "question": "How do I write good test cases for evaluating Claude's outputs?",
        "ground_truth": "Define success criteria first, then build a test set of representative and edge-case inputs with expected outputs or graded rubrics. Prefer many cases over a few, include hard or ambiguous cases, and grade with code (exact/structural checks) or an LLM judge. Automate the suite so you can re-run it as prompts change.",
        "category": "factual",
        "doc": "test-and-evaluate/develop-tests",
    },
    # ============================================================
    # CONCEPTUAL (5) — synthesis across chunks; recall + faithfulness
    # ============================================================
    {
        "question": "What are text embeddings and when would I use them instead of just prompting?",
        "ground_truth": "Embeddings convert text into vectors that capture meaning, so semantically similar text has nearby vectors. Use them for semantic search, retrieval (RAG), clustering, and classification at scale — tasks about comparing or finding similar text — rather than prompting. Anthropic does not provide its own embedding model; the docs recommend third-party providers such as Voyage AI.",
        "category": "conceptual",
        "doc": "build-with-claude/embeddings",
    },
    {
        "question": "What is the difference between extended thinking and adaptive thinking, and when should I use each?",
        "ground_truth": "Extended thinking used a fixed budget_tokens to control reasoning depth. Adaptive thinking (thinking: {type: 'adaptive'}) lets the model decide when and how much to think, tuned with the effort parameter instead of a token budget. Adaptive thinking is recommended on current models; budget_tokens is deprecated on 4.6 and removed on newer models.",
        "category": "conceptual",
        "doc": "build-with-claude/extended-thinking",
    },
    {
        "question": "Why would I count tokens before sending a request, and how do I do it?",
        "ground_truth": "Counting tokens up front lets you estimate cost, stay within the context window, and set max_tokens sensibly. Use the count_tokens endpoint (messages.count_tokens) with the same model you'll call, since counts are model-specific. Don't use generic tokenizers like tiktoken — they mis-count Claude tokens.",
        "category": "conceptual",
        "doc": "build-with-claude/token-counting",
    },
    {
        "question": "How do prompt caching and batch processing each reduce cost, and how do they differ?",
        "ground_truth": "Prompt caching reuses a cached prompt prefix so repeated context is billed at ~10% of input cost — best when many requests share a large stable prefix. Batch processing runs many requests asynchronously at a 50% discount — best for high-volume, non-latency-sensitive jobs. Caching cuts the per-request cost of shared input; batching cuts the price of the whole async job. They can be combined.",
        "category": "conceptual",
        "doc": "build-with-claude/prompt-caching",
    },
    {
        "question": "When should I use tool use versus structured outputs to get machine-readable results?",
        "ground_truth": "Use structured outputs (output_config.format / json_schema) when you just need the model's answer in a guaranteed JSON shape. Use tool use when the model should decide whether and which external function to call, and you'll execute it and feed results back. Tool use is for taking actions or calling code; structured outputs is for constraining the answer's format. Strict tool use combines both with schema-valid tool arguments.",
        "category": "conceptual",
        "doc": "tool-use/overview",
    },
    # ============================================================
    # EXACT-TERM (5) — API params / acronyms; hybrid (BM25) territory
    # ============================================================
    {
        "question": 'What does `cache_control` with `"type": "ephemeral"` do?',
        "ground_truth": "It marks that content block as cacheable with an ephemeral cache entry (default ~5-minute TTL). The marked prefix is written to cache on first use and read back cheaply (~10% of input cost) on subsequent requests that share an identical prefix.",
        "category": "exact_term",
        "doc": "build-with-claude/prompt-caching",
    },
    {
        "question": "What are the possible values of the `tool_choice` parameter?",
        "ground_truth": "auto (model decides whether to use a tool — the default), any (must use some tool), tool with a name (must use that specific tool), and none (cannot use tools). You can also set disable_parallel_tool_use to limit Claude to one tool call per turn.",
        "category": "exact_term",
        "doc": "tool-use/overview",
    },
    {
        "question": "Is the `budget_tokens` parameter still supported?",
        "ground_truth": "budget_tokens (the fixed extended-thinking budget) is deprecated on Claude 4.6 and removed on newer models such as Opus 4.7/4.8 and Fable 5, where sending it returns a 400. Use adaptive thinking (thinking: {type: 'adaptive'}) with the effort parameter instead.",
        "category": "exact_term",
        "doc": "build-with-claude/extended-thinking",
    },
    {
        "question": "What cost discount does the Message Batches API give?",
        "ground_truth": "50% off standard token prices for requests processed through the Batches API.",
        "category": "exact_term",
        "doc": "build-with-claude/batch-processing",
    },
    {
        "question": "What does setting `strict: true` on a tool do?",
        "ground_truth": "It guarantees the tool's input arguments conform exactly to the tool's input_schema — schema-valid structured output for tool calls — rather than best-effort matching.",
        "category": "exact_term",
        "doc": "build-with-claude/structured-outputs",
    },
    # ============================================================
    # CASUAL / MISMATCH (4) — phrased unlike the docs; HyDE/contextual
    # ============================================================
    {
        "question": "Why does Claude keep forgetting what we talked about earlier in a long chat?",
        "ground_truth": "Because the conversation exceeded the model's context window — the maximum number of tokens it can consider at once. Earlier messages fall outside the window (or get compacted/summarized), so they are no longer available. Managing this means compaction, summarization, or trimming old turns.",
        "category": "casual",
        "doc": "build-with-claude/context-windows",
    },
    {
        "question": "My API bill is huge — how do I make it cheaper?",
        "ground_truth": "Reduce token cost: cache large stable prompt prefixes (prompt caching reads at ~10% of input cost), batch non-urgent requests (Batches API gives 50% off), pick a smaller model where it's adequate, trim unnecessary context, and cap max_tokens. Counting tokens first helps find the biggest cost drivers.",
        "category": "casual",
        "doc": "build-with-claude/prompt-caching",
    },
    {
        "question": "How do I stop Claude from making up facts that aren't in my document?",
        "ground_truth": "Ground it in the source: provide the document as a content block with citations enabled so each claim references the document, and instruct the model to answer only from the provided context and to say when the context doesn't contain the answer. Citations let you verify each statement against the source.",
        "category": "casual",
        "doc": "build-with-claude/citations",
    },
    {
        "question": "Can Claude look at a screenshot and tell me what's in it?",
        "ground_truth": "Yes — send the screenshot as an image content block (base64 or URL) with a text question, and Claude can describe or analyze it. Supported formats include PNG, JPEG, GIF, and WebP.",
        "category": "casual",
        "doc": "build-with-claude/vision",
    },
    # ============================================================
    # CANNOT-ANSWER (2) — not in corpus; the model should decline
    # ============================================================
    {
        "question": "What is the exact price per million tokens for Claude Opus 4.8?",
        "ground_truth": "Not answerable from the provided documentation — this corpus covers features, not pricing. The correct response is to decline and say the context doesn't contain pricing information, rather than guess a number.",
        "category": "cannot",
        "doc": None,
    },
    {
        "question": "How do I fine-tune Claude on my own training data?",
        "ground_truth": "Not answerable from the provided documentation — fine-tuning Claude is not covered in this corpus. The correct response is to decline and state that the context doesn't contain this information, rather than fabricate a procedure.",
        "category": "cannot",
        "doc": None,
    },
]


# Quick sanity / breakdown helper.
def category_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in EVAL_QA:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return counts


if __name__ == "__main__":
    print(f"{len(EVAL_QA)} questions")
    for cat, n in category_counts().items():
        print(f"  {cat:11s} {n}")
    docs = {r["doc"] for r in EVAL_QA if r["doc"]}
    print(f"docs covered: {len(docs)}/15")
