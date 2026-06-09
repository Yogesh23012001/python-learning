# Summary Variant Comparison — 2026-06-09

## The hypothesis (pre-committed)

The agentic summary with `fetch_top_repos` should score higher on specificity
and lower on hallucination risk than the legacy summary. Specifically:
- Specificity: agentic > legacy by 0.3+
- Factual grounding: similar (both have access to ground truth)
- Hallucination risk: agentic LOWER than legacy (because of tool grounding)
- Readability: agentic similar or slightly better

## Aggregate scores

Source: `2026-06-09T02-15-16Z_summary_comparison.json` (n=5 examples).

| Variant | n | Specificity | Factual | Halluc | Read | Overall |
|---|---|---|---|---|---|---|
| **legacy** | 5 | 1.00 | 5.00 / 5 | 4.20 / 5 | 5.00 / 5 | **0.95** |
| **agentic_with_top_repos** | 5 | 1.00 | 4.40 / 5 | 3.80 / 5 | 4.80 / 5 | **0.87** |
| Δ (agentic − legacy) | — | 0.00 | **−0.60** | **−0.40** | −0.20 | **−0.08** |

The agentic variant lost ground on **every** dimension except specificity (which was already at the ceiling for both). Hallucination *risk* went the wrong way.

## Per-user comparison

Four of five users scored identically across variants (0.94 each). The divergence sits entirely in one case.

| User | Profile type | Legacy | Agentic | Δ |
|---|---|---|---|---|
| karpathy | famous_ai | 0.94 | **0.61** | **−0.33** ⚠️ |
| torvalds | famous_nonai | 1.00 | 0.94 | −0.06 |
| antirez | known_specialist | 0.94 | 0.94 | 0.00 |
| mojombo | known_historical | 0.94 | 0.94 | 0.00 |
| yyx990803 | famous_oss | 0.94 | 0.94 | 0.00 |

**karpathy side-by-side:**

> **Legacy (factual=5, halluc=4):**
> *"Andrej is an exceptionally prolific and influential developer with 63 public repositories that have collectively garnered 444,791 stars and 64,930 forks... With 197,309 followers and a computed activity score of 633,875.2, he represents a top-tier open-source contributor..."*
>
> Judge reasoning: *"All numerical claims (63 repos, 444,791 stars, 64,930 forks, 197,309 followers, 633,875.2 score) are exactly accurate."*

> **Agentic (factual=2, halluc=3):**
> *"Andrej Karpathy is an exceptionally influential AI/ML developer with 444,791 total stars across 63 public repositories... His most notable projects include **autoresearch** (85,664 stars), **nanoGPT** (59,369 stars), and **nanochat** (54,771 stars)..."*
>
> Judge reasoning: *"The summary accurately reports total stars (444,791), followers (197,309), repos (63), and score (633,875). However, the project names and star counts **appear to be completely fabricated** — there is no way to verify 'autoresearch', 'nanochat', or these specific star counts from the provided data."*

The agentic version was **penalized for using its tool correctly**. The data it cited came from `fetch_top_repos` — real GitHub API output — but the **judge's ground truth was only the aggregate profile** (total_stars, followers, score), not the per-repo data. The judge had no way to verify the repo names, so it defaulted to "fabricated." For users whose top repos the judge happens to know from training (torvalds → linux), it credited the names; for karpathy's specific lesser-known top repos (autoresearch, nanochat), it flagged them as invented.

## Was the hypothesis confirmed?

- [ ] Confirmed
- [ ] Partially confirmed
- [x] **Disconfirmed** — legacy was competitive or better

The legacy variant beat the agentic variant by 0.08 overall (0.95 vs 0.87), and the agentic variant *lost* ground on hallucination risk specifically — the exact dimension the new tool was supposed to *improve*. The hypothesis predicted "Halluc: agentic LOWER than legacy"; the data shows agentic HIGHER (more risk).

That said, the conclusion is fragile. See "What surprised me" below — the result is less about agent quality than about a measurement gap in the eval.

## What surprised me

1. **The judge couldn't verify what the agent fetched.** The `fetch_top_repos` tool returned real repo names from the GitHub API, but the judge's `ground_truth` only contained aggregate stats. The judge had no way to cross-check whether "autoresearch (85,664 stars)" was real or hallucinated, and conservatively marked it as fabricated. **The eval is measuring "what the judge can independently verify," not "what the agent got right."** This is the most important finding in the file — every other interpretation depends on it.

2. **4 of 5 users were ties at 0.94.** With only 5 examples and 4 ties, the entire comparison swings on one case (karpathy). One sample-set perturbation could flip the verdict. Treat the −0.08 aggregate delta as noise-bounded, not load-bearing.

3. **The agent stayed factual on aggregate numbers across both variants.** The judge confirmed 444,791 stars / 197,309 followers / 633,875 score were accurate for the agentic variant. Where the agent went wrong was in attempting *more specificity* — which the eval punished because of the verification gap, not because the agent was actually wrong.

## Honest cost-benefit conclusion

On this eval, the legacy variant wins on score, cost (one LLM call vs an agent loop with extra tool calls), and latency (faster). But the score margin (−0.08) is inside the noise floor of a 5-example dataset, and the agentic variant lost specifically because the judge couldn't see what it fetched. So:

- **Ship legacy when** users want a fast, cheap, low-risk summary that doesn't claim per-project specifics. The judge will reliably score it high because it sticks to verifiable aggregates.
- **Ship agentic when** users want named projects in the summary and you're willing to (a) pay ~2–3× cost/latency and (b) extend the judge's ground truth to include the data the agent's tools fetched, so attempts at richer specificity aren't auto-penalized.

The current setup is the worst of both worlds: the agentic variant pays the agent loop's cost, gets *more* useful information from its tool calls, and then loses points for using it. Either fix the eval (preferred) or stop running the agentic variant on this rubric.

## What I'd improve

1. **Extend the judge's ground truth to include per-repo data.** Pass the `fetch_top_repos` result into `actual_profile_data` so the judge can cross-check named projects against actual API output. Without this, the eval cannot distinguish "the agent used its tool correctly" from "the agent hallucinated." This is the load-bearing fix; nothing else matters until it lands.

2. **n=5 is too few.** Four ties mean a single case (karpathy) drives the verdict. Aim for 15–20 examples spanning profile types (famous, niche, organization, historical), and split the score by `profile_type` to surface where each variant actually wins. With 5 examples you cannot distinguish a real regression from noise.

3. **The third variant was silently dropped.** The script's docstring promises a 3-way comparison ("legacy / agentic_no_top_repos / agentic_with_top_repos") but the actual run only ships two because both agentic variants point at the same endpoint (`/agentic`). To compare "agent without the tool" vs "agent with the tool," either add a `/summary/agentic-minimal` route that doesn't register `fetch_top_repos`, or thread an `enabled_tools` parameter through `run_agent_stream` so each variant can scope its own tool set. As-is, this run cannot answer "is `fetch_top_repos` worth adding to the toolset?" — it can only answer "is agentic worth it overall?"
