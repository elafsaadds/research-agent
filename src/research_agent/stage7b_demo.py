"""Stage 7 evidence: run the research orchestrator on the frozen questions.

Usage:
  python -m research_agent.stage7b_demo               all 8 frozen questions
  python -m research_agent.stage7b_demo 2             the first 2 questions
  python -m research_agent.stage7b_demo Q4 Q8         selected questions
  python -m research_agent.stage7b_demo show RUN_ID   timeline and report of one run
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone

from research_agent.baseline import OUT_DIR, load_questions
from research_agent.model import get_model
from research_agent.policy import Budget
from research_agent.research import (PROMPT_VERSION_RESEARCH, format_research_timeline,
                                     traced_research)
from research_agent.tracing import TraceStore


def run_questions(ids: list[str], limit: int | None) -> None:
    model = get_model()
    budget = Budget.from_env()
    store = TraceStore()
    questions = load_questions()
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    questions = questions[:limit]
    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"stage7_research_{model.name.replace(':', '_')}_{stamp}.jsonl"

    statuses: Counter = Counter()
    original: Counter = Counter()
    total_latency = 0.0
    with path.open("w", encoding="utf-8") as out:
        for q in questions:
            record, res = traced_research(q["question"], model, budget, store)
            statuses[res.status] += 1
            original[res.original_answered] += 1
            total_latency += res.latency_s
            out.write(json.dumps({
                "id": q["id"], "category": q["category"], "run_id": record.run_id,
                "status": res.status, "answered": res.answered, "total": res.total,
                "original_answered": res.original_answered,
                "sub_questions": [s.question for s in res.plan.sub_questions] if res.plan else [],
                "conflicts": [{"sub_question": c.sub_question_id, "sources": c.source_ids}
                              for c in res.conflicts],
                "replans": res.replans, "findings_rejected": res.findings_rejected,
                "tool_calls": res.state.tool_calls, "model_calls": res.model_calls,
                "sources": sorted(res.table.source_ids()) if res.table else [],
                "latency_s": res.latency_s, "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens, "model": res.model,
                "prompt_version": PROMPT_VERSION_RESEARCH, "report": res.message,
            }, ensure_ascii=False) + "\n")
            print(f"{q['id']} {record.run_id} {res.status} coverage={res.answered}/{res.total} "
                  f"original_answered={res.original_answered} conflicts={len(res.conflicts)} "
                  f"replans={res.replans} rejected_findings={res.findings_rejected} "
                  f"tools={res.state.tool_calls} model_calls={res.model_calls} "
                  f"latency={res.latency_s}s")
    n = max(len(questions), 1)
    print(f"\nStatuses: {dict(statuses)}")
    print(f"Original question answered: {dict(original)}")
    print(f"Mean latency: {total_latency / n:.1f}s")
    print(f"Saved: {path}")


def main(argv: list[str]) -> None:
    if argv and argv[0] == "show" and len(argv) == 2:
        rec = TraceStore().load(argv[1])
        print("\n".join(format_research_timeline(rec)))
        print("\n" + (rec.message or ""))
        return
    ids = [a.upper() for a in argv if re.fullmatch(r"[Qq]\d+", a)]
    limit = next((int(a) for a in argv if a.isdigit()), None)
    run_questions(ids, limit)


if __name__ == "__main__":
    main(sys.argv[1:])
