"""Stage 3 evidence: does the model produce parseable decisions?"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from research_agent.baseline import OUT_DIR, load_questions
from research_agent.model import get_model
from research_agent.parsing import request_decision
from research_agent.prompts import PROMPT_VERSION, build_messages


def main(limit: int | None = None) -> None:
    model = get_model()
    questions = load_questions()[:limit]
    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"stage3_parse_{model.name.replace(':', '_')}_{stamp}.jsonl"
    ok_count = first_try = 0
    with path.open("w", encoding="utf-8") as out:
        for q in questions:
            r = request_decision(model, build_messages(q["question"]), max_attempts=3)
            ok_count += r.ok
            first_try += r.ok and r.attempts == 1
            d = r.decision
            record = {
                "id": q["id"], "ok": r.ok, "attempts": r.attempts,
                "status": r.status,
                "action": d.action if d else None,
                "tool_name": d.tool_name if d else None,
                "errors": r.errors, "latency_s": round(r.latency_s, 2),
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "model": model.name, "prompt_version": PROMPT_VERSION,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{q['id']} ok={r.ok} attempts={r.attempts} "
                  f"action={record['action']} tool={record['tool_name']}")
    n = len(questions)
    print(f"\nParsed: {ok_count}/{n} | first attempt: {first_try}/{n}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
