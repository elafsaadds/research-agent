# Build Log: Deep Research Agent (COSC726)

Topic: Agentic AI Security | Pattern: Perplexity-style Deep Research (Level 2)

| Stage | Title | Doc | Status |
|---|---|---|---|
| 0 | Environment and repo skeleton | (this log) | done |
| 1 | Problem, PEAS, success contract | problem_definition.md | done if file exists |
| 2 | Non-agentic baseline | stages/stage_02_baseline.md | done |
| 3 | Structured output + behaviour spec | stages/stage_03_structured_output.md | done |
| 4 | Tools (corpus + arXiv) | stages/stage_04_tools.md | done |
| 5 | Validation gates + bounded loop | stages/stage_05_gates_loop.md | done |
| 6 | Tracing | stages/stage_06_tracing.md | done |
| 7 | Plan, evidence table, conflicts, re-plan | stages/stage_07a_evidence_planning.md, stages/stage_07b_orchestrator.md | in progress (7a, 7b, 7c done); stages/stage_07c_sentence_evidence.md |
| 8 | Human approval gate demonstration | | complete |
| 9A | Research Agent API/UI and model integration | | complete |
| 9B | Security, observability, evaluation and deployment hardening | | planned |

## Decisions
- Framework-neutral: pydantic + httpx.
- Dev model: MockModel. Evaluation model: qwen2.5:3b (Ollama, CPU, 8 GB RAM).

---

## Stage 9A — Research Agent API/UI and Model Integration

**Status:** Completed checkpoint

**Date:** 2026-10-19

**Git branch:** `docs/stage9a-checkpoint`

**Git checkpoint:** `cbf78fd`

Stage 9A introduced the first application-facing interface for the research agent and connected the existing research workflow to a Gradio-based user interface.

### Implemented Components

```text
app/api.py
app/ui.py
src/research_agent/model.py
```

The application API provides a controlled boundary between the user-facing application and the existing research pipeline.

The Gradio interface provides:

```text
Research question input
Research report
Research plan
Evidence table
Detected conflicts
Run diagnostics
```

### Model Integration

The model boundary supports:

```text
Mock
Ollama
Gemini
```

Structured output is controlled through the `STRUCTURED_OUTPUT` environment setting.

When structured output is disabled, the model boundary does not attach the decision schema.

When enabled, the existing decision schema is passed to the supported model provider.

### Verification

The complete automated test suite was executed after the model-boundary correction.

```text
176 passed
```

The test suite includes model-provider, structured-output, agent, policy, evidence, planning, retrieval, tracing, approval, and research workflow tests.

### Research UI Verification

A research question was executed through the user-facing interface:

```text
What are the applications of AI in healthcare?
```

The observed run produced:

```text
Status: PARTIAL
Evidence coverage: 2/3 sub-questions
Model: gemini:gemini-3.1-flash-lite+schema
```

The result demonstrates that the research workflow can expose partial evidence coverage rather than presenting unsupported completeness.

### Scope

Stage 9A established the application/API/UI layer and model-provider integration.

It did not establish production readiness.

Production security hardening, API/UI testing, deployment, and broader evaluation remain Stage 9B work.





