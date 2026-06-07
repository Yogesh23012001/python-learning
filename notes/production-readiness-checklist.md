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