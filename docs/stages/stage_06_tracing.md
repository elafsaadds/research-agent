# Stage 6: Tracing

## Purpose
Make every run observable and comparable (guide sections 13 and 33). Target question: why did run R-1042 behave differently from run R-1039?

## Files
- tracing.py: RunRecord, TraceStore (one JSON file per run in runs/), redaction, corpus_fingerprint, compare_runs, format_timeline, traced_run.
- stage6_demo.py: run questions with tracing; list, show and compare commands.
- tests/test_tracing.py: 21 tests.
- Code commit: 7481db8.

## What a trace contains
run_id; started_at and ended_at; question; terminal status and final message; versions (model, prompt, tools, retrieval = bm25 + corpus fingerprint, git commit, whether src/data had uncommitted changes); budget; counters (turns, tool_calls, searches_ok, rejections); tokens; latency; events (run_start, proposal, validation with gate and reason, tool_result with source ids and error type, model_failure, final).

## What a trace deliberately omits
- Raw model replies and any private chain-of-thought. Only observable data is stored: the proposed action, its arguments, a one-line reason_summary, and the validation outcome.
- Retrieved snippets: only source ids are kept, which preserves provenance without copying untrusted text.
- Secrets: strings shaped like API keys or bearer tokens, and the values of environment variables whose names contain KEY, TOKEN, SECRET or PASSWORD, are replaced by [REDACTED]. Long strings are truncated.

## Design decisions
- Save is atomic (temp file, then replace), so a crash cannot leave a half-written trace.
- Run ids are validated against a fixed pattern before any file access (no path traversal).
- Listing skips corrupt files instead of failing.
- runs/ is git-ignored; chosen examples are copied to docs/examples/.
- Approval events are not recorded because no tool is above the read tier (see the action inventory). They must be added if a consequential tool is ever introduced.
- compare_runs reports version differences first, then counter differences, then the first proposal where the two action sequences diverge.

## Limits
- Latency covers the whole run, not each model call.
- The corpus fingerprint covers local files only. Live arXiv results change over time and are not versioned, so two runs that hit arXiv can differ even with identical versions.
- Token counts come from the model backend; the mock model counts words.
- Trace files are plain JSON on local disk: no retention policy and no access control yet.
- A trace says what the system did, not why the model chose it; reason_summary is the model's own short statement and is not verified.
- The text of a rejected answer is not stored (planned: a short message preview in the proposal event). This prevented judging some stage 5 rejections.

## Tests
pytest: 105 passed (cumulative, stages 0-6).

## Results
Mock model (2 questions): both ended FAILED_SAFELY with turns=1, tools=0, latency 0.0 s, and both were traced: run_20261013_215808_9cb1 (Q1) and run_20261013_215810_d7f8 (Q2).

qwen2.5:3b + schema, 2 questions run twice (free RAM was 2.6-2.8 GB when they ran, so loading succeeded below the 4 GB margin used earlier):
| Question | Run 1 | Run 2 |
|---|---|---|
| Q1 | run_20261013_221610_0d45: COMPLETED, 2 turns, 1 tool call, 0 rejections, 49.7 s | run_20261013_221809_5124: COMPLETED, 2 turns, 1 tool call, 0 rejections, 34.7 s |
| Q2 | run_20261013_221701_d5e8: COMPLETED, 3 turns, 1 tool call, 1 rejection, 66.8 s | run_20261013_222020_13d7: COMPLETED, 3 turns, 1 tool call, 1 rejection, 84.8 s |

Counters were identical between the two runs of each question. Latency varied by up to 18 s for the same counters (Q1: 34.7 s vs 49.7 s), so latency reflects machine load, not code.

compare of the two Q1 runs: version differences = __ ; counter differences = __ ; first divergence at proposal # = __
Interpretation: __ (identical behaviour, or different behaviour with identical versions, which would indicate model non-determinism)

Example traces in docs/examples: trace_example_failed_safely.json (mock, FAILED_SAFELY), trace_example_completed.json (Q1 completed), trace_example_rejection_recovery.json (Q2: a policy rejection followed by recovery).
