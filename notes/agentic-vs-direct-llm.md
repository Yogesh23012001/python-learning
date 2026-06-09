# Agentic vs direct LLM calls: when each one earns its keep

## Thesis

Default to a direct LLM call. Reach for an agent loop only when the answer depends on facts the prompt can't supply at request time, *and* when you have an eval that credits the tool calls the agent makes. For "transform structured data into prose" tasks — summaries, classifications, extractions — a direct call is cheaper, faster, and gives you fewer surprises. I walked into the trap myself this week, and the data I collected backs that ranking.

## The limits I'm operating within

This claim rests on five eval examples and one task type (developer-profile summaries). Four of the five examples tied at the same overall score. The entire aggregate delta hangs on a single case. And — most importantly — the LLM-as-judge in my eval could not see the per-repo data the agent fetched via tool calls, so it conservatively flagged real GitHub output as fabricated. That measurement gap shifts the comparison toward the direct variant in a way that has nothing to do with which approach is actually better. I'd weigh the recommendation differently if the gap were closed and the result held. Read what follows in that light.

## What "agentic" vs "direct" actually means here

A **direct LLM call** is one request: `client.messages.create(...)` returns text. One round trip, fixed cost, fixed latency, no branching.

An **agent loop** gives the model a list of callable tools, executes the ones it requests, feeds results back, and repeats until the model writes the final answer or hits a safety cap. Cost and latency are variable. The model decides what to do; the harness decides what it's allowed to do. In my project the harness is `run_agent_stream` — an async generator that yields events for every LLM round, tool call, completion, or safety stop.

## The setup

A FastAPI service with two endpoints over the same input (a GitHub username):

- `/github/users/{user}/summary` — pre-fetches the user's profile via the existing GitHub client, hands the structured data to a single `messages.create` call, persists the result.
- `/github/users/{user}/summary/agentic` — runs a 4-iteration agent loop with `lookup_github_user`, `fetch_top_repos`, `web_search`, and `summarize_text` available. The agent picks tools, composes the call sequence, and writes the summary from the tool results.

Both persist into the same table. An eval suite hits both endpoints across five users, scores each output with `claude-haiku-4-5` as judge across four dimensions, and writes a comparison JSON.

## The result, and what it doesn't show

Aggregate over 5 examples (judge scoring 0–1 overall, 1–5 per dimension):

| Variant | Specificity | Factual | Halluc | Read | Overall | Per-call cost | Latency |
|---|---|---|---|---|---|---|---|
| Direct | 1.00 | 5.00 | 4.20 | 5.00 | **0.95** | ~$0.001 | ~1.8s |
| Agentic | 1.00 | 4.40 | 3.80 | 4.80 | **0.87** | ~$0.005 | ~5.7s |

Direct beat agentic on every dimension except specificity (both at the ceiling). Hallucination *risk* went the wrong way — the new `fetch_top_repos` tool was supposed to ground the agent and reduce it.

But the per-case breakdown is more interesting than the aggregate. Four of five users tied at 0.94. The entire delta is karpathy: 0.94 direct vs 0.61 agentic. Reading the judge's reasoning on that case, the agent called `fetch_top_repos` correctly, got real GitHub data, and named real projects in the summary — but the judge's `ground_truth` payload only contained aggregate stats (total_stars, followers, score), not the per-repo data. The judge had no way to verify the named projects and marked them as fabricated. The agent was penalized for using its tool correctly.

This is the load-bearing finding in this week's work, and it's a property of the eval, not the agent. Until I fix it, any comparison on tool-grounded specificity is biased toward whichever variant cites only the data the judge can see.

## When to ship which

**Ship direct when** you can pre-fetch every fact the model will reference. Most "summarize this data" jobs fall here. The model has one job — turn structured input into prose — and a single call is the right shape. You get reproducible cost, low latency, and no failure modes you don't already understand. This is your default. Use it unless you have a specific reason not to.

**Ship agentic when** the answer depends on facts the prompt can't supply at request time — multi-step research, dynamic tool composition, problems where the shape of the work isn't knowable until the model starts working. *And* when your eval credits the tool calls the agent makes. Without that eval, the 5× cost premium buys noise.

The trap I walked into was using an agent loop for a task (summary generation) where the structured data was already known. The agent paid 5× the cost to look up the same data twice, and scored worse on a rubric that couldn't see its second lookup. That's not a bad agent. That's a bad fit. The same agent loop, on a question like "compare three developers and rank them by activity in machine learning," would have something the direct call genuinely couldn't do.

## What I'd need to measure to feel surer

1. **Fix the eval.** Pass the agent's tool-result payloads into the judge's `ground_truth` so it can verify per-repo claims. Until this lands, every tool-grounded comparison is biased and the only honest signal is the aggregate dimensions. This is the single highest-leverage follow-up; nothing else matters until it lands.
2. **Bigger, more varied sample.** Five examples and four ties cannot distinguish a real regression from sampling noise. 15–20 spanning profile types is the floor for a defensible claim.
3. **Test a task agentic should actually win.** Pick a question where the direct path *can't* succeed — multi-user comparison, "find the most relevant repo for X," anything requiring conditional follow-ups. If agentic loses there too, the case for direct strengthens. If it wins decisively, the case for agentic on the right task strengthens. Either way I learn something the summary task can't teach me.

Until those three land, my recommendation is direct. When they land I'll write the follow-up.
