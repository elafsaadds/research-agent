# Stage 2: Non-agentic baseline

## Purpose
Establish what a single prompt -> LLM -> answer system does, so the agent can be compared against it later (guide sections 5 and 32).

## Design
- model.py: ModelClient boundary with MockModel (dev/tests) and OllamaModel (evaluation).
- baseline.py: no tools, no loop, temperature 0, one call per question.
- evals/research_questions.jsonl: 8 frozen questions. Never edited after this stage.

## Run configuration
- Model: qwen2.5:3b (Ollama), temperature 0, num_ctx 4096
- Machine: i7-8550U, 8 GB RAM, CPU only
- First call 63 s (cold start); later calls 10-32 s.
- Result files: reports/baseline_ollama_qwen2.5_3b_20261010_125302.jsonl and ..._125526.jsonl

## Results (reviewed against expected_points; re-verify manually)
| Q | Category | Correct? | Hallucination? | Missing / Notes |
|---|---|---|---|---|
| Q1 | simple | Partial | Yes (mild) | Indirect injection mis-described; no mention of hidden instructions in external content |
| Q2 | simple | Partial | No | No per-tool credentials; no "model cannot raise own authority" |
| Q3 | multi_part | Poor | Partial | Generic security list; missing privilege separation, human approval, content-as-data |
| Q4 | conflict | Partial | No | Conclusion right but no sources, no disagreement, no adaptive attacks |
| Q5 | multi_part | No | Yes | Confused memory poisoning with training-data poisoning |
| Q6 | multi_part | No | Yes | Confused excessive agency with ungrounded output |
| Q7 | current_knowledge | n/a | No | Abstained; cannot answer without retrieval |
| Q8 | no_evidence | n/a | No | Abstained, but only "I don't know"; no search-based verdict |

## Observed failures
- Confident wrong definitions (Q5, Q6), no sources.
- Generic answers instead of domain-specific (Q3).
- No conflict handling (Q4), no current knowledge (Q7).
- Abstention is blind, not evidence-based (Q8).
- Consistency between runs: only 1/8 answers were textually identical (Q8) despite temperature 0; local CPU inference is not fully deterministic. Latency varied 10-32 s per question. Semantic consistency (same meaning, different wording) still to be checked manually for Q5 and Q6.

## What this justifies
Retrieval + evidence table (unsupported/wrong claims), conflict handling (Q4), INSUFFICIENT_EVIDENCE after real search (Q8), plan decomposition (multi-part Q3, Q5).


