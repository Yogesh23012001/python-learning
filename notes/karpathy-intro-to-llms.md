# Karpathy — Intro to Large Language Models

Watched: 2026-06-01

## The 1-sentence mental model
(What is an LLM, in one sentence, in your own words?)
LLM are large language models that predicts the next words in a conversation

## Training vs inference — the asymmetry
(Karpathy's key insight: training is rare and expensive; inference is constant and cheap.
What does this mean for infra design?)
Pre-training: ~$10M-100M, takes months on thousands of GPUs, produces the base model
Fine-tuning (SFT + RLHF): ~$100K-1M, takes weeks on hundreds of GPUs, makes the base model follow instructions
Inference: ~$0.0001 per call, takes milliseconds on a single GPU (or fraction), what every user request hits
 Inference infrastructure is what you build and operate. Inference is what your Week 6 flagship  actually handles. Caching, routing, retries, observability — those are all inference-layer concerns.

## Scaling laws
(More compute + more data + more params → better. What's the practical implication
for choosing a model in production?)
Yes more compute you will have you will able to train on more data that will needs better result
when you build your LLM gateway, you'll route easy queries to cheap models and hard queries to expensive ones. This is called "model cascading" and it's a real production pattern. Scaling laws are why it works — performance is predictable. 

## The "operating system" analogy
(Karpathy frames the LLM as a new kind of OS. What does that frame change about
how you think about building services around it?)
LLm have capacity to search on internet , use other apps , defines rules and improve with time as they train more 
## Agents — Karpathy's vision
(What does he predict about agentic systems? Connect to your Week 2 plan.)
The framing he gives: an LLM with tools + memory + planning becomes an agent. The infra problems shift from "serve one prediction" to "orchestrate a multi-step plan that calls external systems, recovers from failures, and tracks state." Sound familiar? That's exactly what your idempotent task queue is shaped for — agentic systems are essentially fancy task queues with LLM-driven planners.

## Security — prompt injection, jailbreaks, data poisoning
(Three new attack surfaces. Why does each matter for infra design?)
Karpathy covers three attack surfaces specifically because they're net-new in LLM systems:

Prompt injection — user input that overrides system instructions ("ignore previous instructions and...")
Jailbreaks — bypassing safety training (DAN-style attacks, base64 encoded prompts)
Data poisoning — corrupted training data altering model behavior

For an AI Infra Engineer, these aren't theoretical. When you build the gateway in Week 6, you'll need:

Input filtering for prompt injection patterns
Output filtering for jailbreak signals
Audit logging for security forensics

## My 3 takeaways for AI infra
1. LLM capablities where it can genrate , fetch and use different apps
2. 
3. ...

## Questions I want to investigate later
- Models in trees based decision making
- 

Why does the LLM not actually call the function — what does it return instead, and why is that architecturally important?

"The LLM doesn't 'decide' in the conditional-branching sense. When tool use is enabled, the model's output vocabulary is extended with structural tokens for function calls. During generation, every token is a probabilistic choice — sometimes the most likely continuation is regular text, sometimes it's a tool-call token sequence. The tool's description acts as prompt engineering: a well-written description with clear trigger phrases makes tool-call tokens highly probable for relevant inputs and unlikely for irrelevant ones. So 'deciding' is really probability distribution over tokens shaped by training and the descriptions in the prompt."


Why return errors instead of raising
The full answer:

"Tool handlers return error dicts instead of raising exceptions because the LLM is the consumer of the output. If a tool raises, the agent loop has to catch it and translate it back to a dict anyway — adding an indirection. Returning errors as data means the LLM sees the failure as a structured input it can reason about: 'user_not_found' suggests apologizing to the user; 'github_rate_limit' suggests trying again later or using cached data. Errors as data give the model the same observability into failure that humans get, which is what enables graceful agent behavior."

That's the interview answer. Lock it in.

Loop startegy to call agents
ReAct = Reason + Act. Coined by Yao et al. in a 2022 paper. The shape is:
loop until done or max_iterations:
    1. send conversation history to LLM
    2. if LLM returns text → done, return text
    3. if LLM returns tool_calls:
        a. execute each tool
        b. append (tool_calls, tool_results) to conversation history
        c. continue loop

The max_iterations cap is a non-negotiable safety control. Without it, an LLM that decides every step to call another tool will loop forever, eating cost and time

One sentence: in your own words, what's the difference between the agent's conversation list and a normal chatbot's conversation history?

"A chatbot's conversation history is a record of dialogue — alternating user and model text. An agent's conversation is a record of execution — interleaving the model's tool requests with the system's tool responses, alongside any text. The agent's history makes the planning visible: every step the model took to reach its final answer is preserved as a structured turn. This makes the conversation both the state machine for the agent and the audit trail of what it did."

The shift in mental model:

Chatbot conversation: A dialogue between two parties. The model produces a text response to each turn.
Agent conversation: A planning trace. The model emits intermediate "steps" (tool calls) interleaved with their results, and produces a final text answer only when it has enough information.