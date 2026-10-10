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
