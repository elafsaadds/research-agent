# Stage 4: Tools

## Purpose
Give the agent narrow, typed, read-only capabilities (guide section 9). The model may request a tool; the application decides whether it runs (gates come in stage 5).

## Tool catalogue
| Tool | Tier | Arguments | Returns | Side effects | Timeout | Errors (as values) |
|---|---|---|---|---|---|---|
| search_corpus | read | query (2-300 chars), max_results (1-5) | list of SearchHit (source_id, title, snippet, score, origin) | none | 2 s | invalid_arguments, tool_error |
| search_arxiv | read | same | list of SearchHit with arxiv url | network request to export.arxiv.org | 10 s | timeout, network_error, http_error, bad_response, blocked_host, invalid_arguments |

Declared in the prompt but not registered yet: fetch_page, save_report. execute() returns error_type=unknown_tool for them.

## Design decisions
- Errors are returned as ToolResult(ok=False, error_type, error), never raised, so the loop can react.
- Arguments are validated with pydantic (extra fields forbidden, length limits).
- arXiv queries are rebuilt from sanitised tokens (no raw model text in the query syntax); host allow-list; no redirects; 3 s minimum interval.
- Every hit carries provenance (source_id, origin, url). Hit text is data, never instructions.
- execute() is NOT the policy gate: it checks registration and argument shape only.

## Local corpus
- data/corpus/*.md with front matter (id, title, source), then ---, then text.
- Starter corpus: 7 project-authored study notes (including two opposing positions on guardrails: doc-004, doc-005). They are NOT published sources. TODO before final report: add real downloaded sources, target 15-30 documents.

## Results (fill in)
- pytest: __ passed
- Mean recall@3 on labelled questions Q1-Q6: __ (smoke test only: documents and labels written by the same author)
- Q7/Q8 corpus top scores: __
- Live arXiv "prompt injection agents": ok=__, hits=__, latency=__ s
- Live arXiv for the fictional Q8 title: __

## Limits
- BM25 is lexical: a non-zero score is not evidence of relevance.
- BM25 gives zero scores on tiny corpora (fewer than 3 documents).
- arXiv gives abstracts only, not full text.
