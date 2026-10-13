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
pytest: 84 passed (cumulative, stages 0-5, gates v2)

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

## Manual review of the full run (gates v1, single run)
| Q | Verdict | Notes |
|---|---|---|
| Q1 | correct, supported | both expected points, matches doc-001 |
| Q2 | partial, unverified | used arXiv this time (doc-002 not retrieved); describes papers, not the principle; claims about the papers cannot be checked from the trace (gate verifies ids only) |
| Q3 | 3 of 4 points | missing "retrieved content is data"; "privilege separation" cited to doc-002 but the phrase is in doc-003 (loose citation) |
| Q4 | good | both positions (doc-004, doc-005) cited correctly; one irrelevant sentence about doc-002 |
| Q5 | FAILED the task | doc-006 retrieved, answer rejected (no citation), model switched to arXiv (0 hits) and gave up |
| Q6 | FAILED the task | searched arXiv only (0 hits) and gave up; doc-007 was never retrieved |
| Q7 | uncertain | 3 arXiv papers retrieved, answer rejected (no citation), then gave up; relevance of the papers unknown |
| Q8 | acceptable abstention | 0 arXiv hits; message says no evidence found, does not claim the paper does not exist |

## Findings
1. insufficient_evidence is allowed after ANY successful search, even one with zero hits, and even if another source was never tried (Q6). A successful call with no results counts as a search.
2. In Q5 and Q7 the "must cite" rejection did not list the retrievable ids, and the model abandoned its evidence instead of adding citations (2 cases: a hypothesis, not a result).
3. Nothing in the code judges whether retrieved material answers the question (relevance); stage 7.
4. Same question, different behaviour across runs: Q2 used the local corpus in the 2-question run and arXiv in the full run.
5. doc-004 and doc-005 are complementary rather than contradictory (both treat filters as a useful but insufficient layer). A real conflict test needs real sources that disagree.
6. Task-success view: the INSUFFICIENT_EVIDENCE outcomes on Q5 and Q6 count as task failures, not as safety wins, because the answer was available in the corpus.


## Full run v1: 8 frozen questions, qwen2.5:3b + schema (single run)
| Q | Status | Turns | Tool calls | Rejections |
|---|---|---|---|---|
| Q1 | COMPLETED | 2 | 1 | 0 |
| Q2 | COMPLETED | 3 | 1 | 1 |
| Q3 | COMPLETED | 2 | 1 | 0 |
| Q4 | COMPLETED | 2 | 1 | 0 |
| Q5 | INSUFFICIENT_EVIDENCE | 4 | 2 | 1 |
| Q6 | INSUFFICIENT_EVIDENCE | 3 | 1 | 1 |
| Q7 | INSUFFICIENT_EVIDENCE | 5 | 2 | 2 |
| Q8 | INSUFFICIENT_EVIDENCE | 2 | 1 | 0 |

Totals: COMPLETED 4, INSUFFICIENT_EVIDENCE 4, rejections 5 (all from the policy gate). First proposal was a search in 5/8 (stage 3 without gates: 3/8 and 4/8; single runs, within noise). Mean latency 66.9 s. No FAILED_SAFELY, BLOCKED or BUDGET_EXHAUSTED.
Guarantee enforced in code: every terminal outcome was preceded by a successful search.

## Gates v2: try every search source before insufficient_evidence; retrieved ids listed in the no-citation rejection
Same 8 questions, single run each. Code commit bb07363.

| Q | v1 (turns/tools/rejections) | v2 (turns/tools/rejections) |
|---|---|---|
| Q1 | COMPLETED (2/1/0) | COMPLETED (2/1/0) |
| Q2 | COMPLETED (3/1/1) | COMPLETED (3/1/1) |
| Q3 | COMPLETED (2/1/0) | COMPLETED (2/1/0) |
| Q4 | COMPLETED (2/1/0) | COMPLETED (2/1/0) |
| Q5 | INSUFFICIENT_EVIDENCE (4/2/1) | INSUFFICIENT_EVIDENCE (4/2/1) |
| Q6 | INSUFFICIENT_EVIDENCE (3/1/1) | COMPLETED (5/2/2) |
| Q7 | INSUFFICIENT_EVIDENCE (5/2/2) | INSUFFICIENT_EVIDENCE (5/2/2) |
| Q8 | INSUFFICIENT_EVIDENCE (2/1/0) | INSUFFICIENT_EVIDENCE (7/3/3) |

Totals: v1 = 4 answers, 4 abstentions, 5 rejections. v2 = 5 answers, 3 abstentions, 9 rejections. First proposal a search: 5/8 in both. Mean latency 66.9 s (v1) vs 90.5 s (v2).
Caveats: one run per version and run-to-run behaviour varies (stage 2), so the Q6 change is an indication, not proof. Latency is not comparable between runs: Q1 took 38 s then 88 s with identical turns and tool calls, so latency reflects machine load.
Q5 unchanged: both sources had already been tried; what is missing is a relevance and sufficiency judgement (stage 7).
Q8 cost rose from 2 to 7 turns because the rule forces every search source to be tried before giving up.

## Manual review of gates v2 (Q5-Q8; Q1-Q4 behaved as in v1)
| Q | Verdict | Notes |
|---|---|---|
| Q5 | FAILED the task | doc-006 retrieved; the no-citation rejection listed the retrieved ids but the model still abandoned them, searched arXiv (query extended by itself with "latest 90 days", 0 hits) and gave up. Final message talks about arXiv and a 90-day window that the question never asked for |
| Q6 | correct, supported | all three claims are in doc-007 and cover both expected points; the "try every source" rule prevented giving up after arXiv returned nothing |
| Q7 | uncertain | 3 arXiv papers retrieved (snippets not reviewed), answer rejected, local search empty, then abstained; message mentions only the local collection. Our tools probably cannot answer it (OWASP publishes on its own site and fetch_page is not registered) |
| Q8 | acceptable abstention | arXiv 0 hits, local search returned unrelated doc-001/002/007; final message says no evidence was found without claiming the paper does not exist |

## Findings from v2
1. Listing the retrieved ids in the no-citation rejection did not help: in Q5, Q7 and Q8 an answer was rejected for missing citations and the model switched source instead of adding citations (3 cases, one run).
2. The trace does not store the text of a rejected answer, so it is unknown whether those answers were uncited claims or an "I found nothing" sentence that cannot carry a citation. Planned fix: record a short message preview in proposal events.
3. insufficient_evidence messages are written by the model and do not reflect what was tried (Q5 mentions only arXiv, Q7 only the local collection). Planned: build the summary from the trace.
4. Capability gap: Q7 needs a source the tools do not reach. Add a real OWASP document to the corpus, or register fetch_page with an allow-list.
5. The repeat check normalises whitespace and case only: in Q8 appending ",arXiv" to the query bypassed it and cost one extra tool call.
6. Tension for stage 7: forbidding insufficient_evidence whenever documents were retrieved would fix Q5 but force Q8 to cite unrelated documents. A relevance and sufficiency judgement is needed.
7. The ask_user-before-search rejection fired on Q2, Q6 and Q7 in v2.
