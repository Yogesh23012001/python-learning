# Week 1 Retro — May 25 to May 31, 2026

A 12-week career-transition roadmap. Week 1 covered Python foundations, async,
FastAPI, Postgres, observability, resilience, and a Postgres-backed task queue
capstone. This document is my honest assessment of what worked, what didn't,
and what changes for Week 2.

---

## Plan vs reality

| Day | Planned hours | Actual hours | What was delivered |
|---|---|---|---|
| Mon | 5 | [your honest #] | URL shortener + Python toolchain |
| Tue | 5 | [#] | Async GitHub fetcher |
| Wed | 5 | [#] | FastAPI + Postgres + Alembic |
| Thu | 5 | [#] | structlog + OTel + Prometheus + circuit breakers |
| Fri | 5 | [#] | LLM integration (Week 2 content pulled forward) |
| Sat | 8 | [#] | Idempotent task queue capstone, v0.1.0 release |
| Sun | 8 | [#] | Karpathy notes + blog post + retro |
| **Total** | **41** | **[your honest total]** | |

**Variance:** [+X / -X hours vs plan. Examples: -8 hours. Or +2 hours. Whichever is true.]

**Why the variance:** [One honest sentence. e.g. "Friday LLM swap consumed Sat capstone time
that we recovered on Sat-Sun rather than skipping" or "Some hours ran ~30% over but I
chose to finish properly rather than cut short."]

---

## What worked

[Fill in 4-6 items. Each item should be a specific thing you can keep doing.]

1. **`mypy --strict` + `ruff` + pre-commit on day 1** — caught type errors that would
   have cost real time on day 3+. Worth the 30-minute Monday setup.
2. **Daily 4-5 hour blocks** — sustainable. No day where I crashed mid-session. Energy
   stayed in the 70-100% zone.

---

## What didn't work

[Be honest. 4-6 items. Don't soften.]

1. **Friday silently swapped from "packaging + Typer CLI" to "LLM integration."** I didn't
   catch the swap; the teacher didn't flag it. We pulled Week 2 content forward without
   discussion. Both repos are stronger for it, but plan integrity was broken.
2. **Saturday day-label confusion** — Hour 5 of Friday was labeled "Saturday Hour 5" and
   cascading confusion followed. Lost ~30 minutes to figuring out what day we were on.
3. **README broken on first push.** Code fences weren't right; ASCII diagrams collapsed in
   GitHub's renderer. I claimed it was fine when it wasn't.
4. **Karpathy notes were thin.** Skipped the agents + security sections. Pushed forward
   instead of going back to absorb them. **This will cost time in Week 2 Monday's tool-use
   hour** because the conceptual scaffolding isn't loaded.

---

## What changes for Week 2

[3-6 specific actions. Not "be more careful" — actual behavior changes.]

1. **Every Monday morning starts with the teacher restating Week N's plan in writing.**
   No more silent swaps. Day-by-day targets visible before any hour begins.
2. **After every commit that affects rendered output (README, diagrams),
   I view the GitHub-rendered version before claiming done.**
3. **For conceptually dense content (Karpathy, agents, RAG, evaluations), I commit
   to filling EVERY note section** — not just the easy ones. "NA" is no longer
   acceptable for foundational mental-model topics.
4. **Add a 1-hour buffer slot on Wednesday** of Week 2 for catching anything that
   slips Mon-Tue. If nothing slipped, use it for blog post #2 draft.
5. **Track actual hours done vs planned in a simple file** updated daily — not just
   in the retro. Drift gets surfaced same-day, not week-end.


---

## What I'd tell myself next Sunday

[One paragraph. Honest, specific, kind. The advice you'd write yourself if you could
time-travel. Don't make it generic. Examples to consider:]

[Example A — if you feel ahead of plan:]
"You did a real week. The capstone is the proof. Don't second-guess what you built;
focus on Week 2's specific gaps — tool use, agent loops, evaluation harnesses. Stop
pulling Week 2 content forward; trust the plan."

---

## Visible artifacts this week

- [`python-learning`](https://github.com/Yogesh23012001/python-learning) — Week 1 main repo (FastAPI + Postgres + observability + LLM integration)
- [`idempotent-task-queue`](https://github.com/Yogesh23012001/idempotent-task-queue) — Week 1 capstone (Postgres-backed task queue)
- [v0.1.0 release](https://github.com/Yogesh23012001/idempotent-task-queue/releases/tag/v0.1.0)
- [Blog post — dev.to](https://dev.to/yogesh23012001/what-a-go-engineer-learns-building-their-first-real-python-service-3b0c)
- [Blog post — Medium](https://medium.com/@yogesh23012001/what-a-go-engineer-learns-building-their-first-real-python-service-f9f6f413951e)
- LinkedIn post — [pending publication]

## Numbers

- **Repos shipped:** 2 (both pinned)
- **Releases:** 1 (v0.1.0)
- **Tests written:** ~30+ (16 in capstone, ~12 in FastAPI tests, ~6 in github_fetcher)
- **Lines of Python:** ~2,500 across both repos
- **Benchmark:** 590 req/s on `GET /tasks` at concurrency 50, p99 228ms
- **Public posts:** 2 published (dev.to, Medium) + 1 pending (LinkedIn)
- **Coverage:** 100% on `python-learning` business logic, 74% on `idempotent-task-queue`

---

## Week 2 target (adjusted)

**Theme:** Agents, tool use, evaluations, prompt caching, guardrails.

**The plan was originally:** LLM fundamentals (tokens, first API calls, structured output).
**The plan is now:** Build on what Week 1 Friday started. Focus on the conceptually-new
content — tool use, agent loops, evaluation harnesses, Anthropic-style prompt caching.

| Day | Theme | Deliverable |
|---|---|---|
| Mon | Tool use / function calling | First LLM-calls-a-Python-function loop, end-to-end |
| Tue | Multi-tool agents + ReAct pattern | Agent that chains GitHub fetcher + DB lookups |
| Wed | Anthropic prompt caching + agent in FastAPI | `/agent/run` endpoint with full tracing |
| Thu | Evaluation harness + LLM-as-judge + guardrails | 20-case eval, automated judging |
| Fri | Upgrade `/summary` to agentic | Agent with eval gates + guardrails |
| Sat | Week 2 capstone — agentic GitHub assistant | Standalone service, optionally deployed to free tier |
| Sun | Embeddings + RAG prep + retro | Notes + Week 3 mental model loaded |

Total: 41 hours planned again. Mid-week buffer slot added to absorb drift.