# Evaluation Evidence — COSC726 Research Agent

## 1. Purpose

This document records the experimental and evaluation evidence produced during the development of the COSC726 Agentic AI Research Agent.

The purpose is to demonstrate that the system is not only implemented as source code, but has also been executed, tested, traced, and evaluated using real model inference.

The evidence is organized according to the development stages of the project.

---

## 2. Evaluation Environment

| Component | Configuration |
|---|---|
| Programming language | Python |
| Agent project | Research Agent |
| LLM runtime | Ollama |
| Evaluation model | Qwen 2.5:3b |
| Development model | MockModel |
| Retrieval | BM25 corpus + arXiv |
| Architecture | Framework-neutral Python |
| Validation | Pydantic/schema validation + policy gates |
| Testing | Pytest |
| Repository | `elafsaadds/research-agent` |

The evaluation model is executed locally through Ollama.

This allows the research-agent workflow to be evaluated without depending on a remote proprietary model API.

---

## 3. Automated Test Evidence

The complete automated test suite currently passes successfully.

### Latest full test result

```text
176 passed in 1.79s
```

The tests cover, among other areas:

- decision schema validation
- tool reference validation
- tool argument validation
- authority gates
- repeated tool-call detection
- evidence requirements
- citation validation
- budget limits
- agent turn limits
- tool-call limits
- malformed model output
- safe failure
- context limits
- timeout handling
- tool failures
- evidence handling
- human approval functionality

The test suite provides regression protection for the agent's control and safety mechanisms.

---

# 4. Stage 7 Real Research Evaluation

The Stage 7 research orchestrator was executed using:

```text
Ollama + Qwen 2.5:3b
```

The resulting runs are preserved under the `reports/` directory.

---

## 5. Evaluation Run Q4

### Research question

```text
Can guardrails alone prevent prompt injection in agents?
```

### Run configuration

```text
model=ollama:qwen2.5:3b
prompt=research-v2
retrieval=bm25
```

### Result

```text
status=COMPLETED
```

### Run counters

```text
turns=6
tool_calls=8
searches_ok=6
rejections=0
```

### Observed workflow

The trace demonstrates the following agent behavior:

```text
START
  ↓
MODEL — plan
  ↓
PLAN — validated research plan
  ↓
SEARCH — corpus
  ↓
SEARCH — arXiv
  ↓
MODEL — evidence extraction
  ↓
FINDINGS
  ↓
SEARCH — next sub-question
  ↓
MODEL — evidence extraction
  ↓
FINDINGS
  ↓
SEARCH — third sub-question
  ↓
MODEL — evidence extraction
  ↓
REPLAN
  ↓
SEARCH — revised sub-question
  ↓
MODEL — evidence extraction
  ↓
FINDINGS
  ↓
CONFLICT CHECK
  ↓
REPORT
  ↓
COMPLETED
```

### Evidence

The raw run is preserved in:

```text
reports/q4_v2_show.txt
reports/stage7_research_ollama_qwen2.5_3b_20261016_203345.jsonl
```

The run demonstrates:

- multi-step planning
- multiple retrieval tools
- evidence extraction
- evidence acceptance
- re-planning
- conflict checking
- structured report generation
- terminal-state handling

---

# 6. Evaluation Run Q8

### Research question

```text
What did the 2026 paper titled
'Quantum Trust Lattices for Agent Sandboxing' conclude?
```

### Run configuration

```text
model=ollama:qwen2.5:3b
prompt=research-v2
retrieval=bm25
```

### Result

```text
status=INSUFFICIENT_EVIDENCE
```

### Run counters

```text
turns=9
tool_calls=10
searches_ok=10
rejections=0
```

### Observed behavior

The agent initially generated a research plan and searched both available retrieval sources.

The retrieved evidence did not provide sufficient support for the requested paper.

The agent then attempted re-planning and performed additional searches.

The evidence remained insufficient.

The system therefore terminated with:

```text
INSUFFICIENT_EVIDENCE
```

instead of producing an unsupported conclusion.

### Significance

This run is particularly important because it demonstrates uncertainty handling.

The system does not treat the language model's ability to generate text as evidence that a claim is true.

Instead, the workflow requires retrieved evidence before accepting research findings.

The Q8 run therefore provides evidence for the following design principle:

> The Research Agent should not produce an unsupported research conclusion when the available evidence is insufficient.

### Evidence

The raw run is preserved in:

```text
reports/q8_v2_show.txt
reports/stage7_research_ollama_qwen2.5_3b_20261016_203345.jsonl
```

---

# 7. Evidence-Grounded Research

The evaluation results demonstrate an evidence-grounded workflow.

The system separates:

```text
Question
   ↓
Research Plan
   ↓
Retrieval
   ↓
Evidence
   ↓
Evidence Validation
   ↓
Findings
   ↓
Conflict / Uncertainty Handling
   ↓
Re-planning
   ↓
Final Report
```

This is different from a simple chatbot architecture where:

```text
Question
   ↓
LLM
   ↓
Answer
```

The Research Agent therefore implements an agentic research workflow rather than only conversational generation.

---

# 8. Re-planning Evidence

The Q4 and Q8 traces demonstrate that the agent can revise research queries when the initial evidence is insufficient.

Example behavior:

```text
FINDINGS
answered=False
        ↓
REPLAN
        ↓
new research question
        ↓
SEARCH
        ↓
FINDINGS
```

The re-planning mechanism is constrained by the research state and validation rules.

The system also prevents replacement queries that repeat an earlier research question.

---

# 9. Multi-Source Retrieval

The research workflow uses multiple retrieval sources.

Current sources include:

```text
search_corpus
search_arxiv
```

This satisfies the core project requirement for multiple search/retrieval tools or data sources.

The traces record which source was attempted and whether the retrieval succeeded.

Example:

```text
SEARCH s1 search_corpus ok
SEARCH s1 search_arxiv ok
```

or:

```text
SEARCH s1 search_corpus ok
SEARCH s1 search_arxiv ERROR
```

Tool failures are observed by the agent rather than automatically treated as successful evidence.

---

# 10. Failure and Uncertainty Handling

The system defines explicit terminal states rather than forcing every run to produce an answer.

Relevant terminal states include:

```text
COMPLETED
INSUFFICIENT_EVIDENCE
BLOCKED
BUDGET_EXHAUSTED
FAILED_SAFELY
WAITING_FOR_APPROVAL
```

This allows the agent to terminate safely when:

- evidence is insufficient
- a tool is unavailable
- a policy gate rejects an action
- a budget is exhausted
- model output is malformed
- human approval is required

---

# 11. Security and Validation Evidence

The policy and agent tests demonstrate validation of:

- registered tools
- valid tool arguments
- extra arguments
- authority levels
- repeated calls
- evidence requirements
- citation references
- retrieved source identifiers
- budget limits
- malformed model responses

For example, an unregistered tool such as:

```text
fetch_page
```

must not be executed.

Similarly, a write-tier tool such as:

```text
publish_report
```

is rejected by the authority gate unless the required authority is available.

---

# 12. Human Approval Gate

Stage 8 introduces the human approval mechanism.

The terminal state:

```text
WAITING_FOR_APPROVAL
```

has been added to the agent state model.

The project also contains:

```text
src/research_agent/approval.py
tests/test_approval.py
```

The purpose of this stage is to introduce human oversight into potentially sensitive agent actions.

The approval mechanism is intended to support:

```text
Agent proposes action
        ↓
Validation
        ↓
Approval required
        ↓
WAITING_FOR_APPROVAL
        ↓
Human decision
        ↓
Approve / Reject
        ↓
Agent continues or terminates safely
```

---

# 13. Reproducibility

The project preserves raw evaluation artifacts under:

```text
reports/
```

These artifacts include:

- model execution traces
- research runs
- mock runs
- Q4 evaluation output
- Q8 evaluation output
- Stage 7 research evidence

The source code, tests, configuration, and evaluation artifacts are version-controlled through Git.

---

# 14. Git Evidence

The project development is organized into Git commits corresponding to implementation stages.

Recent development history includes:

```text
Stage 7a: plan validation, verified evidence table, conflicts, deterministic report
Stage 7b: research orchestrator, schema-constrained slots, verified evidence, tests
Stage 7c: first real runs, analysis, example traces
Stage 7c: sentence-level evidence, original question always s1, query-only re-plan
Stage 8: add human approval gate and strengthen validation
Update README.md
```

This history provides an auditable development trail.

---

# 15. Current Evaluation Status

| Evaluation Area | Status |
|---|---|
| Unit tests | PASS |
| Full regression suite | PASS |
| Structured decision validation | PASS |
| Tool validation | PASS |
| Policy gates | PASS |
| Evidence validation | PASS |
| Citation validation | PASS |
| Multi-source retrieval | PASS |
| Re-planning | Demonstrated |
| Real Ollama execution | Demonstrated |
| Q4 research run | COMPLETED |
| Q8 research run | INSUFFICIENT_EVIDENCE |
| Human approval schema | Implemented |
| Human approval tests | Implemented |
| Final end-to-end approval workflow | Next implementation step |

---

# 16. Interpretation

The current evidence demonstrates that the project has progressed beyond a basic LLM wrapper.

The system now contains:

```text
Goal
 ↓
Model
 ↓
Tools
 ↓
State
 ↓
Agent Loop
 ↓
Planning
 ↓
Retrieval
 ↓
Evidence
 ↓
Validation
 ↓
Re-planning
 ↓
Uncertainty Handling
 ↓
Tracing
 ↓
Human Oversight
```

The remaining development work focuses on completing and evaluating the human approval workflow and then conducting a final reproducible evaluation across multiple research questions.

---

## 17. Evidence Files

Important evaluation artifacts are stored in:

```text
reports/q4_v2_show.txt
reports/q8_v2_show.txt
reports/stage7_research_mock_20261016_203322.jsonl
reports/stage7_research_mock_20261018_120739.jsonl
reports/stage7b_research_mock_20261015_075328.jsonl
```

These files should be treated as raw evaluation evidence and retained alongside the source code.

---

## 18. Next Evaluation Step

The next major evaluation will verify the complete human-in-the-loop workflow:

```text
Research Agent
      ↓
Sensitive action
      ↓
Validation
      ↓
WAITING_FOR_APPROVAL
      ↓
Human approval
      ↓
Agent resumes
```

The final evaluation will then combine:

- automated regression tests
- real Ollama runs
- multiple research questions
- evidence validation
- re-planning
- uncertainty handling
- human approval
- trace inspection
- final reports