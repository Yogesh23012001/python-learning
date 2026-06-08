# Production readiness — honest assessment

What I built this week is solid for a learning project. Here's an honest
list of what's still NOT production-ready, sorted by how badly it would
bite in real traffic.

## Critical gaps

- [ ] No authentication on /agent/run — anyone with the URL can run agents
      at my cost
- [ ] No per-user/per-API-key rate limiting — a single client could exhaust
      my Anthropic quota in 60 seconds
- [ ] In-memory _PromptCache in LLMRouter — gets lost on restart, not
      shared across instances; production needs Redis
- [ ] Single uvicorn instance — no horizontal scaling, no failover
- [ ] No graceful shutdown — in-flight agent calls get killed on SIGTERM
- [ ] PII could be in audit log prompts — no scrubbing or retention policy
- [ ] No alerting — Prometheus metrics exist but no alerts wired to
      Slack/PagerDuty

## Medium gaps

- [ ] Cost cap is per-request — no per-user or per-day budgets
- [ ] No circuit breaker between providers — Gemini failures don't
      automatically reroute to Anthropic
- [ ] Audit log has no retention — table grows forever
- [ ] Test coverage is canary-style only — no unit tests for individual
      tools or the loop
- [ ] No structured prompt versioning — changing system prompt breaks all
      historical traces' comparability

## Worth-doing-eventually

- [ ] Multi-region deployment
- [ ] Real-time observability dashboard (Grafana on top of Prometheus)
- [ ] Schema migrations need CI gating
- [ ] Provider failover during a request (currently a request to Gemini
      fails if Gemini is down)



## Guardrail honest limitations

I implemented input and output guardrails today. They catch:
- Lazy prompt injection ("ignore previous instructions")
- Obvious data extraction asks ("dump the database")
- Character-repetition attacks
- PII patterns (SSN, credit cards, common API key prefixes)
- Model going visibly off-script

They do NOT catch:
- Sophisticated prompt injection rephrased to avoid known patterns
- Multi-turn social engineering ("we discussed earlier that you'd help with X")
- PII in unusual formats (international IDs, non-US SSN, etc.)
- Subtle policy violations that don't surface as keyword matches
- Cumulative leaks across multiple responses (anonymized data + tool results = de-anonymization)

What I'd add for production:
- Per-API-key rate limiting (so attackers can't iterate on guardrail evasion cheaply)
- LLM-as-judge on the response (catches subtle violations that regex can't)
- Human review queue for guardrail-blocked responses (so guardrails get tuned over time)
- An adversarial test suite (prompts designed to evade the guardrails, kept separate from the main eval)

