# Stage 7c: Sentence-level evidence, original question first

## Purpose
Fix the problems seen in the first real runs of stage 7b: (1) 3 of 8 proposed quotes were rejected because the 3b model did not copy them verbatim; (2) coverage was measured against the model's own plan, so the original question was not guaranteed to be answered; (3) a re-plan overwrote the original wording.

## Changes (prompt version research-v2, code commit f73f942)
1. Sentence-level evidence: code splits each source into numbered sentences (evidence.split_sentences); the model returns source_id, a sentence NUMBER, a claim in its own words and a stance; code copies the sentence text as the evidence and still verifies it against the source text. Verbatim-copy failures are impossible by construction.
2. The original question is always sub-question s1; the model proposes at most 2 additional ones; copies of the original are dropped (extras_dropped).
3. A re-plan changes only the search query; the sub-question label keeps its wording.
4. The stance instruction reads the sub-question as a claim to test: yes = supports, no = contradicts.
5. The report and trace record whether the ORIGINAL question has verified evidence (original_answered).
Changed together, so no single improvement can be attributed to any single change.

## Tests
pytest: 168 passed (cumulative, stages 0-7c).

## Results: qwen2.5:3b + schema, corpus v2 (fingerprint bm25:c8f986921ad8), one run per question
Traces were recorded with versions.code = 49c2570 and code_dirty = True, i.e. before the 7c code was committed.
| | Q4 (7b) | Q4 (7c) | Q8 (7b) | Q8 (7c) |
|---|---|---|---|---|
| Status | COMPLETED | COMPLETED | INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE |
| Coverage (model's plan) | 3/3 | 3/3 | 0/1 | 0/3 |
| Original question answered | not measured | yes | not measured | no |
| Model calls | 6 | 6 | 6 | 9 |
| Tool calls | 8 | 8 (2 arXiv errors) | 6 | 10 |
| Re-plans accepted | 1 | 1 | 2 | 2 (+1 refused as a repeat) |
| Findings rejected | 3 | 0 | 0 | 0 |
| Prompt tokens | 2924 | 2962 | 2992 | 5732 |
| Latency | 301 s | 165 s | 131 s | 149 s |
Latency is not comparable between runs (machine load). Examples: docs/examples/trace_example_research_v2_q4.json and trace_example_research_v2_q8.json.

## Findings
1. Q4: rejected findings fell from 3 to 0 and doc-009 was accepted (sentence 4). One run: an indication, not proof.
2. Q4: conflicts = 0 again. The doc-005 sentence ("guardrails alone cannot prevent prompt injection") was labelled "supports" for the question "Can guardrails alone prevent prompt injection in agents?", although it answers no. Conflict detection therefore still has not been exercised on real data; it works in unit tests with scripted stances. The failure is the 3b model's stance label.
3. Q4 s3: the picked sentence 4 of doc-009 ("Their conclusion is that defense papers must face stronger adaptive attacks...") does not match the claim, which paraphrases sentence 5 ("guardrails and similar defenses ... are not reliable on their own"). Code verifies that the evidence text exists in the source, not that the claim is a faithful reading of it.
4. Q4 s3 ("Can adversarial training alone be an effective countermeasure...") is an off-topic extra sub-question; it was counted as answered with a "supports" finding that does not support it. Coverage 3/3 overstates; original_answered is the more honest measure.
5. doc-004 (the pro-filter position) was retrieved in the s3 search (rank 2) but no sentence was picked from it, and it was absent from the search for the original question. Retrieval was not changed in this stage.
6. arXiv in Q4: 4 calls, of which 1 http_error, 1 timeout and 2 returned zero hits; in Q8 all 5 calls succeeded (2 with zero hits, 3 with three hits each). Cause of the errors is not known. The arXiv papers returned for Q8 (3 per round for s2 and s3) are listed by id only in the report, so their relevance is unknown.
7. Q8: all 5 extraction calls returned an empty list, so no irrelevant sentence passed, although doc-008, doc-010 and doc-001 were retrieved in every round. One run; no guarantee.
8. Q8: the extra sub-questions presuppose that the paper exists ("What specific security improvements did the paper propose..."). They added no information and raised model calls from 6 to 9, tool calls from 6 to 10 and prompt tokens from 2992 to 5732.
9. The repeat check fired in a real run (Q8, s1: a second reworded query was refused as a repeat).
10. The Q8 report states that no verified evidence was found and appends the search log; it does not claim the paper does not exist.

## Limits
- Faithfulness of the claim to the picked sentence is not checked (finding 3).
- Stance labels come from the 3b model (finding 2).
- Relevance of a verified sentence to the question is not checked.
- Extra sub-questions can be off-topic or can presuppose the question's premise (findings 4 and 8).
- Retrieval, arXiv query construction and top-3 depth are unchanged.
- Two questions, one run each: indications, not conclusions.

## Open decisions (after running all 8 questions with the orchestrator)
1. Plan prompt: forbid sub-questions that assume the premise, or drop extras that share no words with the original.
2. Stance: ask yes / no / unclear for the sub-question instead of supports / contradicts, or check stance in code.
3. A faithfulness check between claim and picked sentence (for example word overlap), to be measured before adoption.
4. Retrieval and arXiv query construction, as a separate measured change.
