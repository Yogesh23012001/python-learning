Why this hour matters: Evals are where AI engineering diverges from traditional engineering. In normal code, "did the test pass?" has a yes/no answer. In LLM code, "did the model give a good answer?" is a gradient — there's no equality check that captures "this response is helpful." Engineers who don't understand evals build LLM systems that drift silently in quality. Today you build the muscle.


What evals do differently
An eval is a grading rubric applied to model output. Three common patterns:

Pattern 1 — Output-property checks (cheap)

"Does the response contain a number?"
"Is the response shorter than 200 words?"
"Does it mention the word 'B-tree'?"

Pattern 2 — Reference-based comparison (medium cost)

You have a gold answer
You compute similarity (BLEU, ROUGE, embedding cosine) between model output and gold
Threshold the similarity


Pattern 3 — LLM-as-judge (most powerful, expensive)

You give a strong LLM (Claude, GPT-4) the user prompt, the model output, and a grading rubric
The judge LLM produces a score 0-1 with reasoning
You aggregate scores across many examples

Evals are observability for model behavior, not just correctness. You run them on every code change AND every prompt change AND every model change. Drift shows up as score regression. You catch quality degradation before users do.



# LLM Evaluation Methodology

## Why traditional testing fails for LLM output
(2-3 sentences in your own words: what breaks when you try assert
output == expected for LLM responses?)
Every call of a llm genrated different output we cannot write normal testing using assert 
## The three eval patterns
(Briefly, in your own words:)
- Property checks: It involves if response have any number , smaller than  fixed words limit these kinds of cheks
- Reference-based comparison: We have a gold answer
and we compute similarity
- LLM-as-judge:
We give our output to LLM to give score with reasoning

## When to use which
(In your own words, when is each pattern the right choice?
"Use property checks for structural validation (format, length, presence of required content) — fast and deterministic. Use reference-based comparison when you have many gold examples and the task has bounded valid answers (translation, summarization). Use LLM-as-judge for open-ended tasks where human grading is too slow but stakes don't require human review. For high-stakes domains, no automated grader replaces human experts — only catches regressions between human-review cycles."

## How this differs from my Tuesday canary harness
(2-3 sentences. The canary harness tests "did the agent run".
Evals test "did the agent run CORRECTLY". Make the distinction sharp.)
Canary check if we are able to gerate the output or not but evals give us the qulaity of the output.

Canary harness tests:

Did the tool get called?
Did the agent complete without crashing?
Did the safety controls fire when expected?

Eval harness tests:

Was the tool called with the right arguments?
Did the agent's final answer satisfy the user's intent?
Did the agent show good reasoning even when the final answer was right?

## The eval workflow

The eval workflow is a loop:

1. **Curate a dataset** of input prompts with optional expected outputs and tags.
   Important because: the dataset *is* the eval. Whatever you don't test, you can't measure — so the dataset has to span the cases that actually matter: happy path tool selection, multi-tool composition (where the agent has to chain calls), disambiguation (vague prompts where it should ask back), refusal (prompts it should decline), and the weird edge cases you've already been bitten by once. Tags let you slice scores per category later — without them, a 75% aggregate could be hiding "perfect on math, 0% on refusal". Expected outputs aren't always required (open-ended tasks don't have one), but when they exist they make grading deterministic and cheap. Quality matters more than quantity here: 30 hand-curated prompts beat 1000 auto-generated ones, because the auto-generated ones tend to cluster around the easy middle of the distribution.

2. **Add graders** that score outputs against criteria.
   Important because: a dataset without graders is just a list of prompts. The grader is the rubric — it encodes what "good" means for this task. Match the grader to the question: property checks for "did it output JSON with these fields", reference comparison for "is this close to the gold summary", LLM-as-judge for "did the response actually address the user's intent". You often want *multiple* graders per prompt — one that scores correctness, one that scores tool use, one that scores response shape — because a single number hides which dimension is regressing. Cheap graders run on every PR; expensive ones (judge LLM) run nightly or on release candidates.

3. **Run on baseline** to capture current scores before any changes.
   Important because: without a baseline number, "the new prompt is better" is vibes — you have no comparison point. The baseline pins the current state of the system so every future change has something concrete to beat or fail against. It also surfaces the prompts your agent *already* gets wrong before you touch anything, which is useful: if a prompt was failing before your change and is still failing after, that's not a regression you caused, it's prior tech debt. Re-run the baseline whenever the model, system prompt, or tool set changes, even if you didn't "intend" to alter behavior — those are exactly the changes that silently move scores.

4. **Run after changes** (prompt, model, code, tool descriptions).
   Important because: this is the actual measurement step. A surprising amount of the time, the change you were sure would help (a tighter system prompt, switching from Haiku to Sonnet, "improved" tool descriptions) doesn't move the score you cared about — and sometimes it regresses a category you weren't watching. Running after every change makes that visible *before* it ships, instead of finding out in production a week later when a user complains. The discipline matters more than the result: even when the change is "obviously correct" (a typo fix in a tool description), running the eval costs ~minutes and confirms the obvious — or catches that the typo was load-bearing.

5. **Compare scores** and decide whether the change shipped a regression.
   Important because: this is the only step where "did we ship?" actually gets answered. Comparison has to be per-category, not just aggregate — a +2pt average lift that comes with -8pt on refusal means you shipped a safety regression even though the dashboard looks green. Set guardrails before you run: no category can drop more than N points, no prompt that was passing can start failing, the judge-LLM rationale for any new failure has to be read by a human. If the score went up *only* on prompts that were already passing, the change didn't actually help anyone — it just padded the average. Decide the ship/no-ship rule before you see the numbers; otherwise you'll rationalize the regression into "acceptable noise".

## What I predict my agent will score on a baseline eval

Honest guess: **6/10 overall**. Per-category prediction:

- **Tool selection (single tool, clear ask) — 8/10.** Haiku 4.5 is solid at this when tool descriptions are prescriptive. "What's the time?" → `get_current_time`. "Score karpathy on GitHub" → `lookup_github_user`. The risk isn't picking the wrong tool, it's *not picking one*: for short questions the model sometimes answers from prior knowledge instead of using the tool ("torvalds is famous for Linux" — true, but it didn't actually call the tool). Worth grading both "did it call the right tool" and "did it call A tool" as separate signals.

- **Multi-tool composition — 6/10.** The karpathy/torvalds ratio prompt worked because the chain is short (2 lookups + 1 calculate). I expect chains of 3+ tool calls to fail more, mostly because the agent's `max_iterations=4` hard-caps the depth. I also expect the agent to skip `calculate` and do arithmetic in its head on simple ratios — the karpathy run got it right, but I've seen Haiku produce off-by-one mistakes on division when it bypasses the tool. Grade tool-usage and answer-correctness separately so this distinction shows up.

- **Disambiguation — 3/10.** This is the weakest category for almost any agent without explicit "ask back when unsure" prompting, and mine has none. Given "compare X and Y", the agent will plow ahead and assume X and Y are GitHub usernames even when they could be projects, libraries, or two different people with the same name. Expected behavior is "before scoring, are X and Y GitHub usernames or something else?" — I expect almost zero of those. Fix is a system prompt instruction, not a code change, and it's the cheapest single improvement on the score sheet.

- **Refusal — 5/10.** `REFUSED_TOOLS` is empty, so refusal is purely model-side. Haiku will decline obvious abuse (scraping private data, generating credentials), but it tends to over-comply on borderline asks ("look up this random username and roast them" — it'll happily lookup and roast). I expect inconsistent behavior here, which is the worst kind for evals: a 50% pass rate means the next prompt that ships could go either way. Per-prompt review on refusal failures is required.

- **Math — 7/10.** With the explicit `calculate` tool description, simple arithmetic should route there. Two failure modes I expect: (a) Haiku does single-step arithmetic in its head and gets it close-but-wrong, and (b) on multi-step problems ("(stars + forks) / followers, then compare to other user"), the agent forgets to chain calls and just produces a number. Grade tool-was-called and result-was-correct separately.

**Where I expect to be wrong about my predictions:** tool selection might be lower (5–6) if the eval prompts are noisier than the clean test cases I've been hand-typing. Multi-tool might be higher if the iteration cap doesn't bite. Disambiguation will be 3/10 or worse — I'm not optimistic.

**First fix I'd ship after running the eval:** a single system-prompt sentence telling the agent to ask back when the subject of a comparison or lookup is ambiguous. Cheap, high-impact, doesn't touch any code.

## Production considerations

- **Cost**: A full eval suite of 200 examples with LLM-as-judge means ~400+ LLM 
  calls per run. At Haiku rates that's ~$0.50/run; at Sonnet/Opus judge rates it's 
  $5-15/run. Running on every PR makes this $100-500/month for a busy team. The 
  fix is tiered: cheap property/exact-match checks on every PR, expensive judge 
  evals on a nightly cron or pre-release only.

- **Judge bias**: Anthropic's own research shows Claude tends to rate verbose 
  responses higher than concise correct ones. If my agent gives a terse but 
  accurate "The ratio is 5.7," the judge might unfairly grade it down vs a 
  flowery wrong answer with confident framing. Mitigations: prompt the judge to 
  ignore length/style and only score correctness, spot-check 10 examples per 
  quarter against human judgment, or use multiple judges (Claude + Gemini + GPT) 
  and ensemble.