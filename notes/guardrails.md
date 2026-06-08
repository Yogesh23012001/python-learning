Guardrails check the intent and content of the request, not its shape.
The framing I want you to lock in:

Validation rejects malformed requests. Guardrails reject malicious requests. Both can fail; both run before the agent does any real work.


Output guardrails scan the model's response before the user sees it.
What guardrails are NOT
This is the honest part. Guardrails are defense-in-depth, not a complete defense:

A sophisticated attacker will phrase requests to evade your guardrails
Output scanners miss subtle policy violations
Real safety lives in the model's training + your system prompt + audit logging + human review

how do you prevent prompt injection?
I have guardrails as the first line of defense, the system prompt establishes the agent's role, the tool denylist prevents specific actions, audit logging captures attempted attacks for review, and I expect determined attackers to occasionally succeed — that's why the audit log matters.


"Four layers: Pydantic validation on request shape, input guardrails on prompt content, tool denylist on what the agent can call, output guardrails on what it returns. Each catches a different attack class. None alone is sufficient; the combination is the defense. Plus audit logging on every request so guardrail-blocked attempts get reviewed and patterns get tuned."