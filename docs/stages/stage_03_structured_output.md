# Stage 3: Structured output and behaviour specification

## Purpose
Make the model propose decisions in a typed format the application can validate before acting (guide sections 7, 8, 10). The model proposes; the application decides.

## Files
- schemas.py: AgentDecision (tool / answer / ask_user / insufficient_evidence), Finding, SubQuestion, ResearchPlan, TerminalStatus.
- prompts.py: versioned behaviour spec (ROLE, GOAL, TOOLS, RULES, OUTPUT). PROMPT_VERSION = v1. Stable instructions are kept separate from user content.
- parsing.py: tolerant JSON extraction (fences, prose, trailing commas, braces inside strings), schema validation, bounded retry (max 3 attempts) with the rejection reason fed back. Failure ends as FAILED_SAFELY, never as a crash.
- stage3_demo.py: runs the 8 frozen questions and records parse success.

## Key decisions
- Valid JSON is not a valid decision: AgentDecision enforces shape rules (a tool action needs tool_name; other actions need a message; unknown fields are rejected).
- Policy and authority validation is NOT done here; it comes in stage 5.
- Model infrastructure errors are not retried blindly.
- Terminal states defined now: COMPLETED, INSUFFICIENT_EVIDENCE, WAITING_FOR_USER, BLOCKED, BUDGET_EXHAUSTED, FAILED_SAFELY.

## Results (fill after running)
- pytest: __ passed
- Mock model: fails safely after 3 attempts (expected).
- qwen2.5:3b: parsed __/8, first attempt __/8
- Result file: reports/stage3_parse_ollama_qwen2.5_3b_<timestamp>.jsonl
- Common failure reasons observed: (copy from the errors field)

## Limits
- Only structure is checked. A well-formed decision may still be unsafe or wrong; stage 5 adds reference, policy and authority gates.

## Results (single run each)
| Setup | Valid at end | Valid first try | Attempts |
|---|---|---|---|
| prompt only (v1) | 6/8 | 2/8 | 16 |
| + Ollama schema, optional fields | 6/8 | 3/8 | 15 |
| + Ollama schema, two variants (tool / non-tool) | 8/8 | 8/8 | 8 |

Small differences are within run-to-run noise (stage 2: only 1/8 identical answers across two runs).
Cause of the v1 schema failures: the model filled tool_name/arguments while action was not "tool". Constrained output with optional fields cannot express this conditional rule; splitting the schema into two variants makes the invalid mix unrepresentable. AgentDecision validation remains as the second line of defence.

## Key finding: structure is not judgement
With the two-variant schema, only 3/8 first decisions were a search (Q3, Q4, Q8). Q1 and Q5 asked the user for clarification on clear questions; Q2, Q6, Q7 declared insufficient evidence before searching anything.

## Prompt v2 experiment (single run)
Rule added: first action must be a search; ask_user only if truly ambiguous; insufficient_evidence only after a search.
Result: first action was a search in 4/8 (Q3, Q4, Q5, Q8) vs 3/8 with v1, within noise. The failure mode shifted from premature insufficient_evidence (Q2, Q6, Q7) to unnecessary ask_user (Q1, Q2, Q6, Q7).
Conclusion: prompting alone does not enforce search-first on qwen2.5:3b. The rule must be enforced in code by the validation gate (stage 5). Prompt v2 is kept as the stable version.

## Verification
pytest: 26 passed.
Result files: reports/stage3_parse_ollama_qwen2.5_3b+schema_*.jsonl
