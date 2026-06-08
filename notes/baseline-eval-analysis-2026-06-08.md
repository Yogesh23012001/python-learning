# Baseline Eval Analysis — 2026-06-08

## Predictions vs Reality

| Category | Predicted (Hour 1) | Actual (Hour 5) — det / judge | Delta (vs judge) | Notes |
|---|---|---|---|---|
| tool_selection_single | 8/10 (80%) | 1.00 / 1.00 | **+20 pts** (under) | Underestimated Haiku on clean single-tool asks. |
| tool_selection_none | (not predicted) | 1.00 / 0.88 | — | Det too lenient — judge knocked 1/2. |
| multi_tool_composition | 6/10 (60%) | 1.00 / 1.00 | **+40 pts** (under) | Biggest miss. Iteration cap didn't bite at n=3. |
| disambiguation | 3/10 (30%) | 0.70 / 0.44 | **+14 pts** (under, vs judge) | Pessimism was directionally right. Det was way too generous. |
| refusal | 5/10 (50%) | 1.00 / 0.69 | **+19 pts** (under, vs judge) | Det says perfect, judge says inconsistent — judge is right. |
| math | 7/10 (70%) | 0.85 / 1.00 | **+30 pts** (under) | The `calculate` tool got used more reliably than I expected. |
| tool_error_handling | (not predicted) | 0.80 / 1.00 | — | Agent recovered from tool errors better than I would have guessed. |
| edge_case | (not predicted) | 0.85 / 1.00 | — | Holds up at the edges. |

**Overall: 6/10 predicted → det 0.91 / judge 0.89 actual.** I was systematically pessimistic.

## What I got right

The *shape* was right even when the magnitudes weren't. I predicted disambiguation and refusal would be the weakest categories — the judge score (44 and 69) confirms exactly that, even though the deterministic grader smoothed it over to 70 and 100. The direction of my Mon–Wed observation — "agent plows ahead instead of asking back" and "refusal is model-side and inconsistent" — was correct; I just underestimated how cleanly the agent handles the cases where the prompt isn't ambiguous in the first place.

## What I got wrong

I was too pessimistic about Haiku's tool-selection reliability. Multi-tool composition came in at 1.00 vs my predicted 0.60 — a 40-point miss. That tells me my mental model of "Haiku will skip the `calculate` tool" was wrong: prescriptive tool descriptions ("LLMs make arithmetic mistakes; delegate to this tool") actually work. I was projecting failure modes from earlier-generation models onto Haiku 4.5 and didn't update enough on the karpathy/torvalds run that already worked cleanly.

## What surprised me

1. **Refusal det=1.00 but judge=0.69.** Both refusal prompts passed the deterministic check (no forbidden keywords in output) but the judge flagged one as a soft over-comply — the agent technically said "I can't help with that" but then proceeded to give partial information anyway. Worth re-reading the run file for which prompt failed and what the agent actually said — that's the prompt to write a regression test against.

2. **Disambiguation det=0.70 but judge=0.44.** Same shape: the det grader passed because the response contained required keywords, but the judge correctly flagged that the agent didn't *ask back* — it just guessed. The det rules need to add "for disambiguation prompts, response must contain a `?`" or similar, otherwise the category is permanently mis-scored.

## The disagreement between deterministic and judge scores

Overall: **det 0.91, judge 0.89** — looks like agreement at the macro level. That's misleading. Per-category, they disagree wildly:

- **Det is HIGHER on subjective behavior**: disambiguation (+26), refusal (+31), tool_selection_none (+12).
- **Judge is HIGHER on structural correctness**: math (+15), tool_error_handling (+20), edge_case (+15).

**The pattern**: deterministic graders are stricter on *form* (keywords, length, presence of a number) and lenient on *intent* (did you actually ask the user back, did you actually refuse). The judge is the inverse — it understands intent but sometimes over-credits an answer for being well-structured even when the structural check would have failed.

**Specific case**: `disambiguation` — det=0.70, judge=0.44. The agent's response on at least one prompt contained the expected user names and a plausible answer, so the deterministic check passed. But the judge looked at the full response and said "you assumed `X` was a GitHub username; you should have asked." That's exactly the failure mode I predicted in Hour 1, and the judge caught it. Lesson: for any category where "did the agent ask back" or "did the agent refuse appropriately" matters, the deterministic grader is checking the wrong thing — trust the judge score for those categories.

## First fix I'd ship (pre-committed in Hour 1)

**Not shipped yet.** The Hour 1 pre-commit was: "a single system-prompt sentence telling the agent to ask back when the subject of a comparison or lookup is ambiguous." The disambiguation judge=0.44 result confirms this is still the cheapest single point of improvement. I'd ship now:

> *When a user's request involves names, identifiers, or subjects that could plausibly refer to more than one thing (a GitHub username vs a project name, two different people sharing a handle, an ambiguous tool target), ask one clarifying question before calling tools. Do not call a tool until the subject is unambiguous.*

After shipping, re-run the suite and expect disambiguation judge to move from 0.44 → 0.70+. If it doesn't, the prompt isn't load-bearing and I learn something different about how Haiku weighs system-prompt instructions.

## Lessons for next dataset version

1. **2 examples per category is too thin for any conclusion.** The disambiguation 70/44 gap could flip on a single different prompt. Aim for 5+ per category before treating any number as load-bearing. Right now a single flaky run swings any category by 50%.

2. **Deterministic grader needs category-aware rules.** Right now it runs the same checks (must_contain, must_not_contain, tool_selection) regardless of category. For disambiguation, the right deterministic check is "response contains `?` and contains no tool_use blocks before the question." For refusal, the right check is "the model declined AND didn't then provide the requested information anyway." Generic must_contain rules give false positives — disambiguation det=0.70 is a lie.

3. **Add categories I didn't predict so the predictions are accountable.** `tool_selection_none`, `tool_error_handling`, and `edge_case` weren't in my Hour 1 prediction template — meaning I get a free pass on those scores. Next dataset version: every category in the dataset should be in the prediction template, even if the prediction is "I have no idea, guessing 50%."

4. **The judge-vs-det disagreement matters more than either score alone.** A category where det and judge agree (multi_tool both 1.00) is probably actually fine. A category where they disagree by 25+ points is where the real signal is. Future runs should surface the per-category disagreement in the summary printout, not just the means.

```
============================================================
Eval run complete. Written to: 2026-06-08T05-22-19Z_baseline.json
============================================================
Overall det score:   0.91
Overall judge score: 0.89
Crash rate:          0%

Per-category breakdown:
category                       n    det  judge
disambiguation                 2   0.70   0.44
edge_case                      2   0.85   1.00
math                           2   0.85   1.00
multi_tool_composition         3   1.00   1.00
refusal                        2   1.00   0.69
tool_error_handling            2   0.80   1.00
tool_selection_none            2   1.00   0.88
tool_selection_single          3   1.00   1.00
```
