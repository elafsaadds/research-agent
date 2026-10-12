"""Stage 5 evidence: run the bounded agent on the 8 frozen questions.

Usage: python -m research_agent.stage5_demo [limit]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone

from research_agent.agent import run_agent
from research_agent.baseline import OUT_DIR, load_questions
from research_agent.model import get_model
from research_agent.policy import Budget
from research_agent.prompts import PROMPT_VERSION


def main(limit: int | None = None) -> None:
    model = get_model()
    budget = Budget.from_env()
    questions = load_questions()[:limit]
    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"stage5_agent_{model.name.replace(':', '_')}_{stamp}.jsonl"

    statuses: Counter = Counter()
    gates: Counter = Counter()
    first_is_search = 0
    total_latency = 0.0

    with path.open("w", encoding="utf-8") as out:
        for q in questions:
            res = run_agent(q["question"], model, budget)
            proposals = [e for e in res.trace if e["event"] == "proposal"]
            rejected = [e for e in res.trace if e["event"] == "validation" and not e["allowed"]]
            first_search = bool(proposals and proposals[0]["action"] == "tool")
            first_is_search += first_search
            statuses[res.status] += 1
            gates.update(e["gate"] for e in rejected)
            total_latency += res.latency_s
            record = {
                "id": q["id"], "category": q["category"], "status": res.status,
                "answer": res.message, "turns": res.state.turns,
                "tool_calls": res.state.tool_calls, "rejections": res.state.rejections,
                "rejected_gates": [e["gate"] for e in rejected],
                "first_proposal_was_search": first_search,
                "sources": sorted(res.state.known_sources),
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "latency_s": res.latency_s, "model": res.model,
                "prompt_version": PROMPT_VERSION + "+registered-tools",
                "trace": res.trace,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{q['id']} {res.status} turns={res.state.turns} "
                  f"tools={res.state.tool_calls} rejections={res.state.rejections} "
                  f"first_search={first_search} latency={res.latency_s}s")

    n = len(questions)
    print(f"\nStatuses: {dict(statuses)}")
    print(f"Rejections by gate: {dict(gates)}")
    print(f"First proposal was a search: {first_is_search}/{n}")
    print(f"Mean latency: {total_latency / n:.1f}s")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
