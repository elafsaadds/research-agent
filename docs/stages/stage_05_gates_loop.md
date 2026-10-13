# Stage 5: Validation gates and bounded agent loop

## Purpose
Enforce in code what the 3b model does not obey in the prompt (stage 3 finding: only 3-4/8 first decisions were a search). Answers the guide question: which part of my code can say NO to the LLM? (sections 10-12).

## Files
- policy.py: Budget, RunState, validate_decision (gates: reference -> authority -> policy), call_key (canonical form of a call).
- agent.py: run_agent loop with a trace of events; format_observation labels tool output as data.
- stage5_demo.py: runs the 8 frozen questions and records status, rejections by gate, first-proposal-was-search.

## Gates
| Gate | Rejects |
|---|---|
| schema | invalid shape (AgentDecision, stage 3) |
| reference | unregistered tool; invalid or extra arguments |
| authority | tool tier outside allowed_tiers (only read allowed) |
| policy | repeated call (normalised); insufficient_evidence or ask_user before a successful search; answer without evidence, without [source_id] citation, or citing a source never retrieved |

## Loop controls and terminal states
Max turns 8, max tool calls 12, run timeout (RUN_TIMEOUT_SECONDS), context size limit (12000 chars, about 3000 tokens, to stay inside num_ctx 4096), 3 consecutive rejections.
Terminal states: COMPLETED, INSUFFICIENT_EVIDENCE, WAITING_FOR_USER, BLOCKED, BUDGET_EXHAUSTED, FAILED_SAFELY.
The prompt now lists only registered tools (prompt version recorded as v2+registered-tools).

## Known limits
- The timeout is checked between turns; a single model call is not interrupted.
- ask_user is blocked before a search, so a truly ambiguous question is searched first.
- Citation check verifies that the id was retrieved, not that the claim is supported by it (stage 7 and evaluation).
- Trace is an in-memory event list; saved formally in stage 6.

## Tests
pytest: __ passed (gates, loop, budgets, failure modes, tool output labelled as data).

## Results with qwen2.5:3b (fill in)
- Statuses: __
- Rejections by gate: __
- First proposal was a search: __/8 (stage 3 without gates: 3-4/8)
- Mean latency: __ s
- Notes (BLOCKED cases, quality of answers, citation use): __

## Results with qwen2.5:3b (2 questions, single run)
| Q | Status | Turns | Tool calls | Rejections | First proposal a search | Latency |
|---|---|---|---|---|---|---|
| Q1 | COMPLETED | 2 | 1 | 0 | yes | 54.6 s |
| Q2 | COMPLETED | 5 | 2 | 2 (policy) | no | 101.9 s |

Rejections by gate: policy 2. First proposal was a search in 1/2. This sample is too small to compare with the stage 3 rates (3/8 and 4/8).
COMPLETED means the answer passed the gate (its citations point to sources that were retrieved). It does not mean the answer is correct; correctness and claim-to-source support are reviewed manually and evaluated in stage 9.

## Real failure case (documented for the report)
First attempt at these two questions ended FAILED_SAFELY with turns=1 and zero tool calls. Cause: Ollama returned HTTP 500 because it could not allocate a 1.3 GB buffer while loading qwen2.5:3b (8 GB machine, low free RAM). The loop stopped in a named terminal state and the trace recorded the model error. After freeing memory (4.5 GB free before load) the same runs completed.
Not retried automatically: a retry would not help when the cause is memory exhaustion.

## Operational notes
- On this CPU-only machine a turn takes about 10-40 s, so RUN_TIMEOUT_SECONDS must be raised from the default 60 (600 was used here).
- Ollama unloads the model after 5 minutes by default; reloading needs free memory again.
