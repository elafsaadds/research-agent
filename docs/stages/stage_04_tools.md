# Stage 4: Tools

## Purpose
Give the agent narrow, typed, read-only capabilities (guide section 9). The model may request a tool; the application decides whether it runs (gates come in stage 5).

## Tool catalogue
| Tool | Tier | Arguments | Returns | Side effects | Timeout | Errors (as values) |
|---|---|---|---|---|---|---|
| search_corpus | read | query (2-300 chars), max_results (1-5) | list of SearchHit (source_id, title, snippet, score, origin) | none | 2 s | invalid_arguments, tool_error |
| search_arxiv | read | same | list of SearchHit with arxiv url | network request to export.arxiv.org | 10 s | timeout, network_error, http_error, bad_response, blocked_host, invalid_arguments |

Declared in the prompt but not registered: fetch_page, save_report. execute() returns error_type=unknown_tool for them.

## Design decisions
- Errors are returned as ToolResult(ok=False, error_type, error), never raised, so the loop can react.
- Arguments are validated with pydantic (extra fields forbidden, length limits).
- arXiv queries are rebuilt from sanitised tokens (no raw model text in the query syntax); host allow-list; no redirects; 3 s minimum interval.
- Every hit carries provenance (source_id, origin, url). Hit text is data, never instructions.
- execute() is NOT the policy gate: it checks registration and argument shape only.

## Local corpus
- data/corpus/*.md with front matter (id, title, source), then ---, then text.
- Starter corpus: 7 project-authored study notes (doc-004 and doc-005 hold opposing positions on guardrails). They are NOT published sources. TODO before the final report: add real downloaded sources, target 15-30 documents.



**## Results**

- **pytest:** 48 passed.
- **Mean recall@3:** 1.00 over Q1–Q6. This is an initial smoke test; the corpus documents and labels were authored together, so it demonstrates the mechanism works, not retrieval quality.
- **Q7:** No hits; no relevant document was expected.
- **Q8 (fictional paper):** Returned doc-001, doc-002, and doc-007, with a top BM25 score of 0.442. A non-zero score does not establish relevance; Stage 7 needs an explicit sufficiency judgement.
- **Live arXiv query — "prompt injection agents":** `ok=True`, 3 hits, and no reported error. Recorded latencies were 0.89 s and 1.42 s in separate runs.
- **Fictional Q8 title — "Quantum Trust Lattices for Agent Sandboxing":** `ok=True`, `error_type=None`, and zero hits in the direct verification run. A separate run also printed zero hits in 0.655 s. The query completed without a reported tool error, but this does not prove the paper does not exist; it supports returning `INSUFFICIENT_EVIDENCE` when the available evidence is insufficient.

**## Evidence limitations**

The retrieval labels are a small, author-created smoke-test dataset, not an independent evaluation. The fictional-title test confirms zero returned hits for that query at the time of testing; it does not prove global non-existence of the title. Live arXiv results can change over time and are not frozen evidence.

**## Limits**

- BM25 is lexical and can return weakly relevant results due to word overlap.
- BM25 gives zero scores on tiny corpora (fewer than 3 documents).
- arXiv provides abstracts, not full text.
- Live arXiv results change over time, so they are not frozen evidence.

- **pytest:** 48 passed in 1.51s.
- **Mean recall@3:** 1.00 over Q1–Q6. This is an initial smoke test; corpus documents and labels were authored together.
- **Q7:** no hits; no relevant document expected.
- **Q8:** returned doc-001, doc-002, and doc-007; top BM25 score was 0.442. A non-zero score does not establish relevance.
- **Live arXiv query — "prompt injection agents":** `ok=True`, 3 hits, no reported error, latency 0.89s in the first recorded run.
- **Fictional Q8 title — "Quantum Trust Lattices for Agent Sandboxing":** `ok=True`, `error_type=None`, 0 hits in the direct verification run. The query completed without a reported tool error but returned no papers.

## Evidence limitations

The retrieval labels are a small, author-created smoke-test dataset, not an independent evaluation. The fictional-title test confirms zero returned hits for that query at the time of testing; it does not prove global non-existence of the title.
## Limits
- BM25 is lexical and gives zero scores on tiny corpora (fewer than 3 documents).
- arXiv gives abstracts only, not full text.
- Live arXiv results change over time, so they are not frozen evidence.
