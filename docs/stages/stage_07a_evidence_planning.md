# Stage 7a: Plan, evidence table, conflicts, deterministic report

## Purpose
Move correctness out of the 3b model and into code (guide sections 17-19, 21, 44): the model proposes, code verifies and assembles. Stage 5 showed that a loop where the model picks tools fails on relevance and sufficiency (Q5, Q8).

## Files
- retrieval.py: source_text(hit) is the text findings are verified against (full body for corpus hits, abstract for arXiv hits). Stage 7b must show the model exactly this text.
- evidence.py: verify_finding, EvidenceTable, Conflict, validate_conflict, detect_stance_conflicts, render_report, check_report_citations.
- planning.py: validate_plan, fallback_plan, PlanState (progress and bounded re-planning), overall_status.
- schemas.py: new terminal status PARTIAL (some sub-questions have no verified evidence).

## What code verifies
- A finding is accepted only if its source was retrieved for that sub-question and its quote (25-500 characters, whitespace and case ignored) appears verbatim in the source text. Rejected findings are kept with the reason.
- A sub-question counts as answered only with a supporting or contradicting finding; uncertain findings alone do not count.
- Conflicts: detected from stance labels (supports vs contradicts from at least two distinct sources) without a model call; a model-declared conflict must name sources that are in the evidence table.
- Plan: at most 4 sub-questions, no duplicates, none too short, and estimated tool calls must fit the budget. Otherwise a fallback plan (the question itself) is used.
- Re-planning: at most 2 re-plans per run; a replacement may not repeat an earlier wording for the same sub-question (loop detection).
- Report: assembled from the evidence table only; every claim line carries a [source_id]; check_report_citations flags any unknown id. Brackets and pipes inside quotes are neutralised so quoted text cannot forge a citation or break the table.

## What code cannot verify (limits)
- That the claim is a faithful reading of the quote, or that the quote answers the sub-question. This stays with the model and is measured by manual review.
- A quote that is real but irrelevant would pass. A 25-character minimum stops trivial quotes only.
- Stance labels come from the model; conflict detection is only as good as those labels.

## Tests
pytest: __ passed (cumulative). New: test_evidence.py, test_planning.py, test_retrieval.py.

## Next: stage 7b
Orchestrator that calls the model (plan, extraction, re-plan wording), runs both sources per sub-question through execute, integrates tracing, and evaluates on the 8 frozen questions against the stage 5 loop and the baseline.
