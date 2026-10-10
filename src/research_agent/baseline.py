"""Non-agentic baseline: question -> prompt -> LLM -> answer. No tools, no loop."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from research_agent.model import ModelError, get_model

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "evals" / "research_questions.jsonl"
OUT_DIR = ROOT / "reports"

SYSTEM = "You are a research assistant. Answer the question concisely."


def load_questions() -> list[dict]:
    with QUESTIONS.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_baseline() -> Path:
    model = get_model()
    OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = model.name.replace(":", "_")
    out_path = OUT_DIR / f"baseline_{safe_name}_{stamp}.jsonl"

    with out_path.open("w", encoding="utf-8") as out:
        for q in load_questions():
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": q["question"]},
            ]
            try:
                resp = model.generate(messages)
                record = {
                    "id": q["id"], "category": q["category"],
                    "question": q["question"], "answer": resp.text,
                    "model": resp.model, "latency_s": round(resp.latency_s, 2),
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "error": None,
                    "review": {"correct": None, "hallucination": None,
                               "missing": None, "notes": ""},
                }
            except ModelError as exc:
                record = {"id": q["id"], "category": q["category"],
                          "question": q["question"], "answer": None,
                          "error": str(exc)}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{q['id']} [{q['category']}] done")
    print(f"\nSaved: {out_path}")
    return out_path


if __name__ == "__main__":
    run_baseline()
