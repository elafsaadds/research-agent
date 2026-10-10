# Build Log: Deep Research Agent (COSC726)

Topic: Agentic AI Security | Pattern: Perplexity-style Deep Research (Level 2)

| Stage | Title | Doc | Status |
|---|---|---|---|
| 0 | Environment and repo skeleton | (this log) | done |
| 1 | Problem, PEAS, success contract | problem_definition.md | done if file exists |
| 2 | Non-agentic baseline | stages/stage_02_baseline.md | done |
| 3 | Structured output + behaviour spec | stages/stage_03_structured_output.md | done |
| 4 | Tools (corpus + arXiv) | stages/stage_04_tools.md | done |
| 5 | Validation gates + bounded loop | | |
| 6 | Tracing | | |
| 7 | Plan, evidence table, conflicts | | |
| 8 | Critic + re-plan (stretch) | | |
| 9 | Security, evals, API/UI, Docker | | |

## Decisions
- Framework-neutral: pydantic + httpx.
- Dev model: MockModel. Evaluation model: qwen2.5:3b (Ollama, CPU, 8 GB RAM).


