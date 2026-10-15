# Stage 7b: Research orchestrator

## Purpose
Replace the stage 5 loop, where the 3b model picks tools and decides when to stop, by a pipeline where code drives and the model fills three narrow slots (guide sections 15-19, 21, 44). Stage 5 showed that the model fails on relevance and sufficiency (Q5, Q8).

## Files
- research.py: run_research, traced_research, format_research_timeline, prompts, JSON schemas.
- stage7b_demo.py: python -m research_agent.stage7b_demo Q4 Q8 (or a number, or no argument for all 8); "show RUN_ID" prints a timeline and the report.
- tests/test_research.py: 15 tests with a scripted model, no network.

## Pipeline
1. Plan (model): 1-3 sub-questions. validate_plan checks count, duplicates, length and tool budget; otherwise the fallback plan (the question itself) is used and the rejection is counted.
2. Search (code): every sub-question is searched in BOTH sources through execute. The model does not choose tools.
3. Extraction (model, schema-constrained): findings = source_id, claim, quote, stance. Code accepts a finding only if the source was retrieved for that sub-question and the quote (25-500 chars) appears verbatim in the text shown to the model (each source truncated to 1200 chars, and verification uses the same truncated text).
4. Re-plan (model, bounded): a sub-question without a supporting or contradicting verified finding is reworded, at most MAX_REPLANS (2) times per run; a wording already tried is refused.
5. Conflicts (code): stance supports vs contradicts from at least two distinct sources.
6. Report (code): built only from the evidence table; every claim has a [source_id]; a citation check runs on the final text; a search log is appended. Status: COMPLETED, PARTIAL, INSUFFICIENT_EVIDENCE, BUDGET_EXHAUSTED or FAILED_SAFELY.

## Budgets and controls
Model calls (MAX_AGENT_TURNS; every attempt counts), tool calls (MAX_TOOL_CALLS), run timeout (checked between steps), context size (sources dropped from the end of the list if the prompt is too large), 2 attempts per model slot. A model server failure ends the run as FAILED_SAFELY. Tool output reaches the model labelled as data.

## Trace
Same RunRecord as stage 6, prompt version research-v1. Events: run_start, plan, model_call, search, findings, replan, conflicts, budget, model_failure, report, final. Counter mapping: turns = model calls; rejections = findings rejected by verification plus plans rejected. Raw model replies and retrieved text are not stored; quotes are truncated to 200 characters in findings events.
Examples: docs/examples/trace_example_research_q4.json and trace_example_research_q8_abstention.json. Both were recorded with versions.code = a292413 before the 7b code was committed.

## Tests
pytest: 160 passed (cumulative, stages 0-7b).

## Results: qwen2.5:3b + schema, corpus v2 (10 documents, fingerprint bm25:c8f986921ad8), one run per question
| Q | Status | Coverage | Model calls | Tool calls | Re-plans | Findings rejected | Latency |
|---|---|---|---|---|---|---|---|
| Q4 | COMPLETED | 3/3 | 6 | 8 | 1 | 3 | 301 s |
| Q8 | INSUFFICIENT_EVIDENCE | 0/1 | 6 | 6 | 2 | 0 | 131 s |
Not comparable with stage 5 results: different corpus (v1 vs v2) and one run each. Model time was about 299 of 301 s (Q4) and 127 of 131 s (Q8).

## Findings
1. Verification works and is needed: 3 of 8 proposed findings (5 accepted) were rejected as non-verbatim quotes (doc-010 once, doc-009 twice). Cause not examined: the trace keeps the rejection reason, not the rejected text. Cost: the doc-009 evidence was lost twice.
2. Q4: conflicts = 0. Visible in the trace: doc-004 (pro-filter position) was never retrieved; doc-009 was retrieved twice but its quotes were rejected; doc-010 was retrieved every round but used once, for an unrelated sub-question; doc-005 was labelled "supports" for the "limitations of guardrails" sub-question, so no finding was labelled "contradicts". Conflict detection was therefore not exercised on real data; it works in unit tests with scripted stances.
3. Coverage is relative to the model's own plan. Q4 was COMPLETED with 3/3, yet the first sub-question (types of injection attacks) is off the core of the question, and of the three expected_points only two appear (adaptive attacks bypass filters; layered defence); "the sources disagree" is missing. The guide asks for an independent coverage check against the original task requirements.
4. A real but irrelevant quote passed: the MELON line under "types of injection attacks" is verbatim from doc-010 but describes a defense. Verified quote does not mean relevant quote.
5. The report has no direct yes/no conclusion; it lists findings per sub-question.
6. Q8: the model returned an empty findings list three times although doc-008 and doc-010 appeared in every round, so the feared failure (a real but irrelevant quote) did not occur in this run (one run; no guarantee). The report says no verified evidence was found, appends the search log, and does not claim the paper does not exist. The two re-plans reworded the same meaning ("key findings", "main findings"), costing 4 tool calls and about 80 s of model time without new information; the cap of 2 bounded the waste.
7. arXiv returned zero hits for 4 of 7 queries. Hypothesis, not verified: the query builder ANDs up to six tokens, which is too strict for long sub-questions.

## Limits
- Code checks that a quote exists in the source, not that the claim reads it faithfully, and not that the quote answers the sub-question.
- Stance labels come from the 3b model; conflict detection is only as good as they are.
- A re-plan rewrites the sub-question in place; the report shows the final wording.
- Retrieval is unchanged from stage 4 (BM25 top 3 per source).
- Two questions, one run each: indications, not conclusions.

## Planned changes (stage 7c)
1. The model picks sentence ids from numbered source sentences and code copies the exact text as the evidence, so verbatim-copy failures disappear by construction (faithfulness of the claim to the sentence stays unchecked).
2. Always include the original question as a guaranteed sub-question, so coverage is checked against the original task and not only the model's plan.
3. Separate, measured change: arXiv query construction and retrieval depth.
