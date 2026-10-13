"""Stage 6 evidence: traced runs, plus tools to show and compare them.

Usage:
  python -m research_agent.stage6_demo [run] [limit]    run questions with tracing
  python -m research_agent.stage6_demo list             list recent runs
  python -m research_agent.stage6_demo show RUN_ID      print one run's timeline
  python -m research_agent.stage6_demo compare A B      why did two runs differ?
"""
from __future__ import annotations

import sys
from collections import Counter

from research_agent.baseline import load_questions
from research_agent.model import get_model
from research_agent.policy import Budget
from research_agent.tracing import (TraceStore, compare_runs, format_timeline,
                                    traced_run)


def run_questions(limit: int | None) -> None:
    model = get_model()
    budget = Budget.from_env()
    store = TraceStore()
    statuses: Counter = Counter()
    for q in load_questions()[:limit]:
        rec = traced_run(q["question"], model, budget, store)
        statuses[rec.status] += 1
        print(f"{q['id']} {rec.run_id} {rec.status} turns={rec.counters['turns']} "
              f"tools={rec.counters['tool_calls']} rejections={rec.counters['rejections']} "
              f"latency={rec.latency_s}s")
    print(f"\nStatuses: {dict(statuses)}")
    print(f"Traces saved in: {store.dir}")


def main(argv: list[str]) -> None:
    cmd = argv[0] if argv else "run"
    store = TraceStore()
    if cmd == "list":
        for r in store.list():
            print(f"{r['run_id']} {r['status']:<22} turns={r['turns']} "
                  f"tools={r['tool_calls']} {r['latency_s']}s | {r['question']}")
    elif cmd == "show" and len(argv) == 2:
        print("\n".join(format_timeline(store.load(argv[1]))))
    elif cmd == "compare" and len(argv) == 3:
        d = compare_runs(store.load(argv[1]), store.load(argv[2]))
        print(f"same question: {d['same_question']} | status: {d['status']}")
        print(f"version differences: {d['version_diffs'] or 'none'}")
        print(f"counter differences: {d['counter_diffs'] or 'none'}")
        print(f"first divergence at proposal #: {d['first_divergence']}")
        print("A:", *d["sequence_a"], sep="\n  ")
        print("B:", *d["sequence_b"], sep="\n  ")
    else:
        limit = None
        rest = argv[1:] if cmd == "run" else argv
        if rest and rest[0].isdigit():
            limit = int(rest[0])
        run_questions(limit)


if __name__ == "__main__":
    main(sys.argv[1:])
