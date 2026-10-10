"""Stage 4 evidence: does search_corpus find the labelled documents?

Usage:
  python -m research_agent.stage4_demo                                  # corpus check
  python -m research_agent.stage4_demo arxiv prompt injection agents    # one live arXiv query
"""
from __future__ import annotations

import json
import sys

from research_agent.baseline import ROOT, load_questions
from research_agent.tools import execute


def corpus_check(k: int = 3) -> None:
    labels = json.loads((ROOT / "evals" / "retrieval_labels.json").read_text(encoding="utf-8-sig"))
    recalls = []
    for q in load_questions():
        r = execute("search_corpus", {"query": q["question"], "max_results": k})
        got = [h.source_id for h in r.hits]
        want = labels.get(q["id"], [])
        if want:
            recall = len(set(got) & set(want)) / len(want)
            recalls.append(recall)
            print(f"{q['id']} want={want} got={got} recall@{k}={recall:.2f}")
        else:
            print(f"{q['id']} (no relevant doc expected) got={got} "
                  f"top_score={r.hits[0].score if r.hits else None}")
    if recalls:
        print(f"\nMean recall@{k} over {len(recalls)} labelled questions: "
              f"{sum(recalls) / len(recalls):.2f}")


def arxiv_check(query: str) -> None:
    r = execute("search_arxiv", {"query": query, "max_results": 3})
    print(f"ok={r.ok} error_type={r.error_type} error={r.error} latency={r.latency_s}s")
    for h in r.hits:
        print(f"- {h.source_id} | {h.title}\n  {h.snippet[:120]}...")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "arxiv":
        arxiv_check(" ".join(sys.argv[2:]))
    else:
        corpus_check()
