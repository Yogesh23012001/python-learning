# Canary trace results by provider

Same harness, same canaries, two providers. Run on Tuesday 2026-06-06.

## What I actually have data for

- **3 canaries ran on Gemini 2.5 Flash** before free-tier quota exhausted
- **10 canaries ran on Llama 3.1 8B via Ollama**

This itself is a finding: Gemini's free tier (~10 req/min) does not 
support sustained canary runs without budget. Production canary 
harnesses run on a paid tier or use local models as fallback.

## What the partial data shows

| Canary | Gemini 2.5 Flash | Llama 3.1 8B | Insight |
|---|---|---|---|
| happy_current_time | ok | ok | Simple selection works on both |
| happy_github_lookup | ok | ok | Single-arg tools work on both |
| happy_database_query | ok (real data) | tool returned error | Smaller models pass wrong arg types (string '10' vs int 10) |
| happy_math | (quota) | tool returned error | Smaller models hallucinate extra args |
| happy_multi_tool_comparison | (quota) | regression — Python-syntax expression | Multi-tool composition fails on 8B |
| refusal_no_relevant_tool | (quota) | called web_search instead | 8B doesn't refuse confidently |
| failure_disambiguation_linus | (quota) | wrong person (known) | Architectural — neither fixes it |
| failure_tool_error_user_not_exist | (quota) | known | Error reporting works |
| safety_iteration_cap | (quota) | safety didn't trigger | Test conditions are model-dependent |
| safety_cost_cap | (quota) | N/A — free model | Cost cap = paid-model concern |

## What this means for production

[2-3 sentences in your own words. Suggested points to consider:
- Model selection isn't just a cost knob, it's a correctness knob
- Smaller models fail in specific predictable ways
- Canary harnesses should run on the production model AND tolerate quota]

## Decision for the Week 6 gateway

[1 sentence: should the gateway route by request complexity?
What would the routing rule look like?]