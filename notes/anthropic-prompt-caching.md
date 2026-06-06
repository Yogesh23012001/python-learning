# Anthropic Prompt Caching


About Promt Caching is it really useful ?
"Prompt caching shifts input cost from per-token to per-write. For agentic workloads where the same system prompt and tool definitions are sent on every iteration, this can reduce input cost by 60-90% depending on iteration depth. The breakeven is two cache hits — anything that gets reused at least twice within the cache TTL is economically worth caching. Production AI gateways treat cache management as a first-class concern, not an optimization."

Prompt caching is the highest-leverage cost knob in production AI. Not model selection. Not output token limits. Cache management. Teams that don't think about caching pay 3-5x more than they need to. The Week 6 gateway's caching layer is the single highest-value feature it will have.

The cache-point placement is a design decision. Putting it after tool definitions is the default-good choice. Putting it after the first message of a conversation is the better-for-multi-turn choice. Putting it after every static system instruction is overkill. You'll make these trade-offs explicitly when building.

Cache deployment is operationally non-trivial. When you change a tool description, every agent run for the next 5+ minutes gets a cache write (1.25x cost) instead of a cache read (0.1x cost). Tool description changes are deployments, not refactors. Senior teams treat them with deploy-style caution.

## The 1-sentence mental model
(In your own words: what does prompt caching do, mechanically?)
Prompt caching help us to reduce our cost for api calls , It stores of prompts in cache upto a marker if our next prompt match the cached prompt we get charged inly 10 per of normal input token costr
## What gets cached
(Describe in your own words what content qualifies for caching and how
you mark a cache point. The key insight is "prefix must match exactly
up to the breakpoint" — explain why this matters.)
 We cached the system prompt and tool prompts.

## The pricing model
(Three multipliers — write cost, read cost, normal cost. Where does
breakeven sit? Why is 2 hits the magic number?)
Anthorpic use fenrally 1.25 time of the input cost for first input in case of cached conversation . next will. charged as .1 of cost . this brings math where if we have atleast two calls that it will leads us to cost saving in comparison to normal conversation

## My agent's expected savings
(Estimate using your actual workload. Pick one of your canary prompts.
Roughly how many tokens of system+tools do you send? How many iterations?
What's the percent savings?)

For a  more complex promt total token will be around 10000 , and expected iteration around 3-4 , 
we can save upto 30-40 per of cost in these cases


## What this changes about how I think about agent design
(One paragraph. Before today you'd optimize for "fewer iterations" to
save cost. With caching, the calculus shifts — what's now worth
optimizing for? What's now cheaper to do?)
Now we should see where the prompt caching can also play a role along with less iteration , if there is a high oteration call it should definately have the prompt caching done to reduce cose

## Production considerations / what could go wrong
(Three honest items: things that could make caching not help, or even hurt.)
1. if we choose caching for less iteration , it will lead to more cost bcz intial input token cost is high is  case of caching 
2. Not hitting the cache breakpoint
3. tool description changes break the cache for every concurrent user, not just yours.



"LLM inference has two phases: prefill, where the model processes the prompt left-to-right and builds up KV cache state — this is quadratic in prompt length and dominates input cost — and generation, where output tokens are produced relatively cheaply from that state. Prompt caching saves the KV cache to fast storage after the first request and reuses it on subsequent requests with matching prefixes. The model skips the expensive prefill work and loads the pre-computed state instead. The 10% pricing on cache hits reflects the actual ratio of storage-load cost to compute cost."