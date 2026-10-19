# Research Agent

A bounded, evidence-grounded research agent developed for the COSC726 Agentic Artificial Intelligence project.

The system is inspired by the workflow of Perplexity-style Deep Research systems. Instead of acting as a simple chatbot, the project progressively implements an agent architecture with structured decisions, tool use, validation gates, bounded execution, evidence tracking, research planning, source verification, tracing, evaluation, and human-approval controls.

## Project Goal

The goal of this project is to build a research agent that can:

1. Receive a research question.
2. Decompose or plan the research task.
3. Select appropriate retrieval tools.
4. Search multiple sources.
5. Observe retrieved evidence.
6. Maintain research state.
7. Validate model-generated actions before execution.
8. Prevent unsafe or unauthorized tool calls.
9. Track evidence and source provenance.
10. Produce citation-grounded answers.
11. Handle insufficient evidence and tool failures safely.
12. Operate within bounded execution limits.
13. Record traces and experimental evidence.
14. Demonstrate a human approval gate for protected actions.

The implementation follows the principle:

> The LLM is not the agent. The agent is the controlled system around the model.

---

# Project Context

**Course:** COSC726 — Agentic Artificial Intelligence

**Project Pattern:** Perplexity-style Deep Research

**Agent Level:** Level 2 — Applied Agent

**Research Pattern:** Research & Knowledge Agent

**Primary Architecture:**

```text
Goal
  |
  v
Model
  |
  v
Structured Decision
  |
  v
Validation Gates
  |
  +------> Reject / Feedback ------+
  |                                |
  v                                |
Tool Execution                     |
  |                                |
  v                                |
Observation -----------------------+
  |
  v
Research State
  |
  v
Evidence / Planning / Re-planning
  |
  v
Citation-Grounded Answer
```

---

# Development Stages

The project has been developed incrementally so that each stage adds a specific agent capability and corresponding evidence.

| Stage | Capability | Status |
|---|---|---|
| 0 | Environment and repository skeleton | Complete |
| 1 | Problem definition, PEAS and success contract | Complete |
| 2 | Non-agentic baseline | Complete |
| 3 | Structured output and behavior specification | Complete |
| 4 | Retrieval tools: local corpus + arXiv | Complete |
| 5 | Validation gates and bounded agent loop | Complete |
| 6 | Tracing and execution observability | Complete |
| 7a | Planning, evidence table and conflict handling | Complete |
| 7b | Research orchestrator and verified evidence | Complete |
| 7c | Sentence-level evidence and query-only re-planning | Complete |
| 8 | Human approval gate demonstration | Complete as standalone demonstration |
| 9 | Security, evaluation expansion, API/UI and deployment | Future work |

---

# Current Implementation

The current repository contains the following major components.

## Agent Loop

`src/research_agent/agent.py`

Implements the bounded control loop:

```text
Observe
   |
Decide
   |
Validate
   |
Act
   |
Observe
   |
Repeat until terminal state
```

The agent is constrained by:

- Maximum number of turns
- Maximum tool calls
- Maximum execution time
- Maximum context size
- Maximum consecutive validation rejections

The loop produces a structured `AgentResult` containing:

- Terminal status
- Final message
- Run state
- Execution trace
- Prompt token count
- Completion token count
- Latency
- Model name

---

# Model Boundary

`src/research_agent/model.py`

The model is accessed through a controlled model interface rather than being allowed to directly control the application.

The current local development environment uses Ollama.

Example model:

```text
qwen2.5:3b
```

Earlier development also used:

```text
qwen2.5:1.5b
```

The model boundary exposes structured information such as:

- Generated text
- Model name
- Latency
- Prompt tokens
- Completion tokens

This separation makes it possible to use mock models for deterministic tests and Ollama for local evaluation.

---

# Retrieval Tools

`src/research_agent/tools.py`

The current retrieval layer contains two read-only search tools.

## 1. Local Corpus

```text
search_corpus
```

The local corpus is searched using BM25 retrieval.

## 2. arXiv

```text
search_arxiv
```

The tool retrieves research papers from arXiv through its API.

The tools share typed result structures:

```text
SearchHit
ToolResult
```

Each retrieved result contains provenance information such as:

- Source ID
- Title
- URL when available
- Snippet
- Retrieval score when applicable
- Source origin

---

# Tool Safety

Tools are registered explicitly in a tool registry.

Each tool has:

```text
name
tier
description
argument schema
function
timeout
side effects
```

The current agent allows only:

```text
read
```

tools through the authority policy.

This prevents the model from directly invoking unregistered or unauthorized operations.

---

# Structured Decisions

`src/research_agent/schemas.py`

The model does not freely control program execution.

Instead, it produces a structured `AgentDecision`.

Typical actions include:

```text
tool
answer
insufficient_evidence
ask_user
```

The structured decision is validated before execution.

This creates a separation between:

```text
Model proposal
      |
      v
Application validation
      |
      v
Execution
```

---

# Validation Gates

`src/research_agent/policy.py`

The policy layer is one of the main safety boundaries of the system.

The validation order is:

```text
Schema
  |
Reference
  |
Authority
  |
Policy
```

A proposal can therefore be rejected before any tool is executed.

Examples of rejected behavior include:

- Unknown tool
- Invalid tool arguments
- Extra arguments
- Unauthorized tool tier
- Repeated tool call
- Answer without evidence
- Answer without citations
- Citation to a source that was never retrieved
- Insufficient-evidence decision before required searches

A rejection is recorded as data and fed back into the agent loop.

---

# Bounded Execution

The agent does not run indefinitely.

Default controls include:

```text
MAX_AGENT_TURNS = 8
MAX_TOOL_CALLS = 12
RUN_TIMEOUT_SECONDS = 60
MAX_REJECTIONS = 3
```

The system can terminate safely with states such as:

```text
COMPLETED
INSUFFICIENT_EVIDENCE
WAITING_FOR_USER
WAITING_FOR_APPROVAL
BLOCKED
BUDGET_EXHAUSTED
FAILED_SAFELY
```

---

# Evidence and Citations

The research workflow maintains retrieved sources in the agent state.

Answers are required to cite retrieved source IDs.

For example:

```text
The retrieved evidence indicates that indirect prompt injection
can place untrusted instructions inside external content [doc-001].
```

The policy layer verifies that cited source IDs actually exist in the retrieved evidence.

This prevents the model from fabricating arbitrary citations.

---

# Research Planning and Orchestration

The project contains dedicated modules for research planning and orchestration:

```text
planning.py
research.py
retrieval.py
evidence.py
memory.py
```

The Stage 7 development introduced a more structured research workflow including:

- Research plans
- Query decomposition
- Evidence slots
- Verified evidence
- Source provenance
- Conflict handling
- Re-planning
- Sentence-level evidence
- Research reports

The Stage 7 work also includes recorded mock and Ollama research runs in the `reports/` directory.

---

# Tracing

`src/research_agent/tracing.py`

Agent execution is observable through structured trace events.

Examples include:

```text
run_start
proposal
validation
tool_result
model_failure
final
```

Tracing makes it possible to inspect:

- What the model proposed
- Which validation gate accepted or rejected it
- Which tools were executed
- What evidence was retrieved
- Why the run terminated

This is important for debugging and academic evaluation of agent behavior.

---

# Human Approval Gate

`src/research_agent/approval.py`

The project includes a standalone human-approval demonstration for a protected action.

The simulated action is:

```text
simulate_publish_research_report
```

The demonstration supports three decisions:

```text
pending
rejected
approved
```

Expected terminal states are:

```text
pending  -> WAITING_FOR_APPROVAL
rejected -> BLOCKED
approved -> COMPLETED
```

The operation is intentionally simulated and does not publish anything externally.

### Important implementation boundary

The current approval component is a **standalone demonstration** of the human-approval control.

It is not yet fully integrated as a mandatory transition inside the main `run_agent()` execution loop.

Integrating the approval gate into the production research workflow is a planned next step.

---

# Testing

The project uses `pytest`.

The latest complete test run produced:

```text
176 passed in 2.05s
```

Run the complete test suite with:

```powershell
pytest -q
```

Individual test modules can also be executed.

For example:

```powershell
pytest tests\test_agent.py -q
```

```powershell
pytest tests\test_policy.py -q
```

The agent tests cover behavior including:

- Normal tool → answer flow
- Insufficient evidence handling
- Repeated tool calls
- Unknown tools
- Turn limits
- Tool-call limits
- Tool failures
- Malformed model output
- Context limits
- Timeout limits
- Prompt-injection-style tool output
- Fabricated citations
- Source coverage
- Trace generation
- Registered-tool prompting

The policy tests cover:

- Tool reference validation
- Argument validation
- Extra-argument rejection
- Authority enforcement
- Repeat-call detection
- Evidence requirements
- Citation validation
- Environment-based budgets
- Search-source requirements

The approval tests verify:

- Pending approval
- Rejected approval
- Approved simulated action

---

# Experimental Evidence

The `reports/` directory contains recorded experimental artifacts from different development stages.

Examples include:

```text
baseline_*.jsonl
stage3_parse_*.jsonl
stage5_agent_*.jsonl
stage7b_research_*.jsonl
stage7_research_*.jsonl
q4_v2_show.txt
q8_v2_show.txt
```

These files provide execution evidence from mock models and local Ollama experiments.

The reports should be interpreted as experimental records rather than as proof that every future deployment scenario has been evaluated.

---

# Project Structure

```text
research-agent/
|
+-- src/
|   +-- research_agent/
|       +-- __init__.py
|       +-- agent.py
|       +-- approval.py
|       +-- baseline.py
|       +-- evaluation.py
|       +-- evidence.py
|       +-- memory.py
|       +-- model.py
|       +-- parsing.py
|       +-- planning.py
|       +-- policy.py
|       +-- prompts.py
|       +-- research.py
|       +-- retrieval.py
|       +-- schemas.py
|       +-- tools.py
|       +-- tracing.py
|       +-- stage3_demo.py
|       +-- stage4_demo.py
|       +-- stage5_demo.py
|       +-- stage6_demo.py
|       +-- stage7b_demo.py
|
+-- tests/
|
+-- reports/
|
+-- docs/
|   +-- build_log.md
|
+-- pyproject.toml
+-- uv.lock
+-- requirements.txt
+-- README.md
+-- .env.example
+-- .gitignore
```

---

# Installation

Create and activate the project environment.

Example:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies using the project's dependency configuration.

If using `uv`:

```powershell
uv sync
```

The repository also contains an exported:

```text
requirements.txt
```

generated from the locked dependency configuration.

---

# Ollama Setup

Install Ollama and make sure the Ollama service is running.

Check installed models:

```powershell
ollama list
```

Example:

```text
qwen2.5:3b
```

Run the model manually if required:

```powershell
ollama run qwen2.5:3b
```

The local Ollama API normally runs at:

```text
http://localhost:11434
```

---

# Environment Configuration

The project supports environment configuration through `.env`.

An example configuration can be based on:

```text
.env.example
```

Typical local model configuration includes:

```text
MODEL_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:3b
```

---

# Running Tests

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run:

```powershell
pytest -q
```

Expected current result:

```text
176 passed
```

The exact execution time can vary by machine.

---

# Git Development History

The repository uses incremental commits to document the development process.

Important milestones include:

```text
Stage 7a
Plan validation, verified evidence table, conflicts, deterministic report

Stage 7b
Research orchestrator, schema-constrained slots, verified evidence

Stage 7c
Sentence-level evidence, original question handling, query-only re-planning

Stage 8
Human approval gate and strengthened validation
```

The latest Stage 8 commit is:

```text
f982945
Stage 8: add human approval gate and strengthen validation
```

The Git history is intentionally maintained so that the evolution of the agent architecture can be inspected rather than presenting only a final snapshot.

---

# Current Status

## Implemented

- Local LLM model boundary
- Ollama integration
- Structured model decisions
- Local BM25 corpus retrieval
- arXiv retrieval
- Typed tool registry
- Tool argument validation
- Authority checks
- Policy validation gates
- Bounded agent loop
- Tool-call budgets
- Turn budgets
- Runtime timeout
- Context limits
- Safe failure handling
- Evidence tracking
- Citation validation
- Research planning
- Research orchestration
- Evidence verification
- Conflict handling
- Re-planning
- Sentence-level evidence
- Structured tracing
- Experimental reports
- Automated test suite
- Human approval gate demonstration

## Current Verification

```text
176 tests passed
```
## Not Yet Fully Integrated

- Human approval as a mandatory transition inside the main research loop
- Production deployment
- Docker deployment
- Larger-scale benchmark evaluation
- Extended security evaluation
- Production-grade API hardening
- Production-grade UI hardening



# Limitations

The current implementation is a research/academic prototype.

Important limitations include:

1. The research sources are limited to the implemented local corpus and arXiv retrieval tools.
2. Local Ollama performance depends on the available hardware and model.
3. The current human approval mechanism is a standalone simulation.
4. The evaluation dataset is still limited compared with a production research system.
5. Retrieval quality depends on the available corpus and search queries.
6. The system does not claim production-level reliability or comprehensive factual correctness.
7. Additional security testing is required before deployment in consequential environments.

---

# Future Work

Planned improvements include:

1. Integrate the human approval gate directly into the main agent workflow.
2. Add a critic/reviewer agent for intermediate research results.
3. Implement automatic re-planning after critic rejection.
4. Expand retrieval sources.
5. Improve evidence ranking and sentence-level attribution.
6. Expand the evaluation set beyond the current research questions.
7. Add API/UI interfaces.
8. Add Docker-based deployment.
9. Conduct broader security and prompt-injection testing.
10. Evaluate the system using reproducible research-agent benchmarks.

---

# Academic Evidence

The repository is structured to preserve both implementation and development evidence.

The main evidence sources are:

```text
src/research_agent/
    Implementation

tests/
    Automated verification

reports/
    Experimental execution records

docs/build_log.md
    Development-stage record

Git history
    Incremental implementation evidence
```

This structure allows the project supervisor to inspect not only the final implementation but also the progression from a basic local LLM integration toward a bounded, evidence-grounded agent architecture.

---

# Repository

GitHub repository:

https://github.com/elafsaadds/research-agent

Branch:

```text
main
```

Latest published Stage 8 commit:

```text
f982945
```
---

# Development Progress

| 8 | Human approval gate demonstration | Complete as standalone demonstration |
| 9A | Research Agent API/UI and model integration | Complete |
| 9B | Production hardening, security, evaluation and deployment | In progress |

**Date:** 2026-10-19

### Git Checkpoint

The current implementation has been preserved in GitHub on:

```text
Branch: docs/stage9a-checkpoint
Commit: cbf78fd

### Stage 9A — Research Agent UI and Model Integration

**Status:** Completed checkpoint

**Date:** 2026-10-19

Stage 9A introduced the first working user-facing interface for the research agent and completed the model-provider integration required by the research workflow.

#### Implemented Components

The following application components were added or completed:

```text
app/
├── api.py
└── ui.py
```

`app/api.py` provides the application-facing interface to the existing research pipeline.

`app/ui.py` provides a Gradio-based web interface that allows a user to submit a research question and inspect the resulting research artifacts.

The interface exposes:

```text
Research Report
Research Plan
Evidence Table
Detected Conflicts
Run Diagnostics
```

#### Model Integration

The model boundary in:

```text
src/research_agent/model.py
```

supports multiple model providers through the same `ModelClient` abstraction.

The implemented providers include:

```text
Mock
Ollama
Gemini
```

Structured output can be enabled through:

```text
STRUCTURED_OUTPUT=1
```

When structured output is enabled, the model receives the project's decision schema and returns structured JSON suitable for validation by the existing agent pipeline.

When structured output is not enabled, the model boundary does not attach a decision schema.

This behavior is covered by automated tests.

#### Gemini Verification

Gemini was successfully connected and tested through the model boundary.

The verified configuration used:

```text
MODEL_PROVIDER=gemini
MODEL_NAME=gemini-3.1-flash-lite
STRUCTURED_OUTPUT=1
```

A direct model test successfully returned a structured response and confirmed connectivity.

The research UI was subsequently tested using the Gemini model with structured output enabled.

#### Research UI Verification

The Gradio interface successfully executed a research question:

```text
What are the applications of AI in healthcare?
```

The resulting run demonstrated:

```text
Status: PARTIAL
Model: gemini:gemini-3.1-flash-lite+schema
Coverage: 2/3 sub-questions
```



Stage 9A focused on:

1. User-facing research interface
2. Application API boundary
3. Gemini model integration
4. Structured model output configuration
5. UI-to-research-pipeline integration
6. Research result visualization
7. , evidence-grounded architecture.---

---

# Development Progress

| Stage | Description | Status |
|---|---|---|
| 8 | Human approval gate demonstration | Complete as standalone demonstration |
| 9A | Research Agent API/UI and model integration | Completed checkpoint |
| 9B | UI observability, security, evaluation, and hardening | Planned |

**Documentation review date:** 2026-10-19

## Stage 9A — Research Agent UI and Model Integration

**Status:** Completed checkpoint

**Git branch:** `docs/stage9a-checkpoint`

**Checkpoint commit:** `cbf78fd`

### Implemented Components

The application includes:

- `app/api.py` — application-facing interface to the research pipeline.
- `app/ui.py` — Gradio interface for submitting research questions and inspecting results.
- `src/research_agent/model.py` — model-provider integration.

The implemented model providers include Mock, Ollama, and Gemini.

Structured output can be enabled with the `STRUCTURED_OUTPUT=1` environment setting.

The interface exposes research artifacts including the report, research plan, evidence table, detected conflicts, and run diagnostics.

### Research UI Verification

A healthcare research question was tested through the interface:

`What are the applications of AI in healthcare?`

The recorded result was partial, with evidence coverage for two of three sub-questions. This demonstrates a partial research outcome in that run; it does not establish general performance or guarantee complete evidence coverage.

### Automated Tests

The previously reported test result was **176 passed**. The test suite must be run again to verify the current working tree before the next checkpoint.

### Scope and Limitations

Stage 9A established the application interface and connected it to the research workflow. It did not establish production readiness.

## Stage 9B — Planned Work

Stage 9B will be scoped after reviewing the current implementation and existing tests. Planned areas for investigation include:

1. Review UI progress and status reporting.
2. Improve visibility of sub-question execution and evidence coverage.
3. Review error handling and diagnostic information.
4. Add or improve API/UI tests where needed.
5. Review security and configuration handling.
6. Record verified results in the project documentation.

A capability will only be marked complete after implementation and verification.