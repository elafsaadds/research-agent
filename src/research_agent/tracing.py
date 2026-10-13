"""Tracing: observable evidence for every run (guide sections 13 and 33).

Records what the system did and observed: proposals, validation outcomes, tool
results (source ids only), counters, terminal status, tokens, latency, versions.
It does NOT record private chain-of-thought or raw model replies.
Approval events do not exist yet: all tools are read-only (see the action inventory).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from research_agent.agent import AgentResult, run_agent
from research_agent.model import ModelClient
from research_agent.policy import Budget
from research_agent.prompts import PROMPT_VERSION
from research_agent.tools import CORPUS_DIR, ROOT

TOOLS_VERSION = "tools-v1"
RUNS_DIR = ROOT / "runs"
RUN_ID_RE = re.compile(r"^run_\d{8}_\d{6}_[0-9a-f]{4}$")
MAX_STR = 2000

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
]
_SENSITIVE_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.I)


# ---------- redaction ----------

def redact_text(text: str, limit: int = MAX_STR) -> str:
    for value in [v for k, v in os.environ.items()
                  if _SENSITIVE_NAME.search(k) and len(v) >= 8]:
        text = text.replace(value, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    if len(text) > limit:
        text = text[:limit] + f"...[truncated {len(text) - limit} chars]"
    return text


def redact(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    return obj


# ---------- versions ----------

def corpus_fingerprint(directory: Path | None = None) -> str:
    """Short hash of the local corpus, so a run can name the exact data it searched."""
    directory = Path(directory) if directory is not None else CORPUS_DIR
    if not directory.exists():
        return "none"
    h = hashlib.sha256()
    files = sorted(directory.glob("*.md"))
    if not files:
        return "none"
    for p in files:
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def _git_info() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, timeout=5)
        if commit.returncode != 0:
            return {"commit": None, "dirty": None}
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "data"],
                               cwd=ROOT, capture_output=True, text=True, timeout=5)
        return {"commit": commit.stdout.strip() or None,
                "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


# ---------- record ----------

class RunRecord(BaseModel):
    run_id: str
    started_at: str
    ended_at: str
    question: str
    status: str
    message: str | None = None
    versions: dict[str, Any]
    budget: dict[str, Any]
    counters: dict[str, int]
    tokens: dict[str, int]
    latency_s: float
    events: list[dict[str, Any]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"run_{now:%Y%m%d_%H%M%S}_{secrets.token_hex(2)}"


def build_record(result: AgentResult, question: str, budget: Budget, run_id: str,
                 started_at: str, ended_at: str, prompt_version: str) -> RunRecord:
    b = asdict(budget)
    b["allowed_tiers"] = sorted(b["allowed_tiers"])
    git = _git_info()
    s = result.state
    return RunRecord(
        run_id=run_id, started_at=started_at, ended_at=ended_at,
        question=redact_text(question),
        status=result.status,
        message=None if result.message is None else redact_text(result.message, 4000),
        versions={"model": result.model, "prompt": prompt_version,
                  "tools": TOOLS_VERSION, "retrieval": "bm25:" + corpus_fingerprint(),
                  "code": git["commit"], "code_dirty": git["dirty"]},
        budget=b,
        counters={"turns": s.turns, "tool_calls": s.tool_calls,
                  "searches_ok": s.searches_ok, "rejections": s.rejections},
        tokens={"prompt": result.prompt_tokens, "completion": result.completion_tokens},
        latency_s=result.latency_s,
        events=redact(result.trace),
    )


# ---------- store ----------

class TraceStore:
    def __init__(self, directory: Path | None = None):
        self.dir = Path(directory) if directory is not None else RUNS_DIR

    def _path(self, run_id: str) -> Path:
        if not RUN_ID_RE.match(run_id):
            raise ValueError(f"invalid run id: {run_id!r}")
        return self.dir / f"{run_id}.json"

    def save(self, record: RunRecord) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(record.run_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)  # atomic: no half-written trace
        return path

    def load(self, run_id: str) -> RunRecord:
        path = self._path(run_id)
        return RunRecord.model_validate_json(path.read_text(encoding="utf-8-sig"))

    def list(self, limit: int = 20) -> list[dict]:
        if not self.dir.exists():
            return []
        out = []
        for p in sorted(self.dir.glob("run_*.json"), reverse=True):
            try:
                r = RunRecord.model_validate_json(p.read_text(encoding="utf-8-sig"))
            except (ValueError, OSError):
                continue  # skip corrupt files instead of failing the listing
            out.append({"run_id": r.run_id, "started_at": r.started_at,
                        "status": r.status, "question": r.question[:60],
                        "turns": r.counters.get("turns"),
                        "tool_calls": r.counters.get("tool_calls"),
                        "latency_s": r.latency_s})
            if len(out) >= limit:
                break
        return out


# ---------- run, show, compare ----------

def traced_run(question: str, model: ModelClient, budget: Budget | None = None,
               store: TraceStore | None = None,
               prompt_version: str = PROMPT_VERSION + "+registered-tools") -> RunRecord:
    budget = budget or Budget.from_env()
    store = store or TraceStore()
    run_id = new_run_id()
    started = _now_iso()
    result = run_agent(question, model, budget)
    record = build_record(result, question, budget, run_id, started, _now_iso(),
                          prompt_version)
    store.save(record)
    return record


def format_timeline(rec: RunRecord) -> list[str]:
    v = rec.versions
    lines = [f"{rec.run_id} status={rec.status} model={v.get('model')} "
             f"prompt={v.get('prompt')} retrieval={v.get('retrieval')} "
             f"code={v.get('code')} latency={rec.latency_s}s",
             f"counters={rec.counters} tokens={rec.tokens}"]
    for e in rec.events:
        t = f"[{float(e.get('t', 0)):7.2f}s]"
        kind = e.get("event")
        if kind == "run_start":
            lines.append(f"{t} START   {str(e.get('question', ''))[:90]}")
        elif kind == "proposal":
            args = json.dumps(e.get("arguments"), ensure_ascii=False) if e.get("arguments") else ""
            lines.append(f"{t} PROPOSE turn={e.get('turn')} {e.get('action')} "
                         f"{e.get('tool') or ''} {args} (attempts={e.get('attempts')})")
        elif kind == "validation":
            if e.get("allowed"):
                lines.append(f"{t} VALIDATE allowed")
            else:
                lines.append(f"{t} VALIDATE REJECTED [{e.get('gate')}] {e.get('reason')}")
        elif kind == "tool_result":
            status = "ok" if e.get("ok") else f"ERROR {e.get('error_type')}"
            lines.append(f"{t} TOOL    {e.get('tool')} {status} hits={e.get('hits')} "
                         f"({e.get('latency_s')}s)")
        elif kind == "model_failure":
            lines.append(f"{t} MODEL FAILURE after {e.get('attempts')} attempts: {e.get('errors')}")
        elif kind == "final":
            lines.append(f"{t} FINAL   {e.get('status')} turns={e.get('turns')} "
                         f"tool_calls={e.get('tool_calls')} rejections={e.get('rejections')}")
    return lines


def action_sequence(rec: RunRecord) -> list[str]:
    """One short string per proposal; a rejection is appended to its proposal."""
    seq: list[str] = []
    for e in rec.events:
        if e.get("event") == "proposal":
            query = (e.get("arguments") or {}).get("query") or "-"
            seq.append(f"{e.get('action')}|{e.get('tool') or '-'}|{query}")
        elif e.get("event") == "validation" and not e.get("allowed") and seq:
            seq[-1] += f" -> REJECTED[{e.get('gate')}]"
    return seq


def compare_runs(a: RunRecord, b: RunRecord) -> dict:
    """Answer: why did run A behave differently from run B?"""
    keys = sorted(set(a.versions) | set(b.versions))
    version_diffs = {k: (a.versions.get(k), b.versions.get(k))
                     for k in keys if a.versions.get(k) != b.versions.get(k)}
    sa, sb = action_sequence(a), action_sequence(b)
    first = None
    for i in range(max(len(sa), len(sb))):
        if (sa[i] if i < len(sa) else None) != (sb[i] if i < len(sb) else None):
            first = i
            break
    return {
        "same_question": a.question == b.question,
        "status": (a.status, b.status),
        "version_diffs": version_diffs,
        "counter_diffs": {k: (a.counters.get(k), b.counters.get(k))
                          for k in sorted(set(a.counters) | set(b.counters))
                          if a.counters.get(k) != b.counters.get(k)},
        "first_divergence": first,
        "sequence_a": sa,
        "sequence_b": sb,
    }
