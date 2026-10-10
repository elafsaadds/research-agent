"""Tolerant parser: small local models often wrap JSON in prose or code fences."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from research_agent.model import ModelClient, ModelError
from research_agent.schemas import AgentDecision, TerminalStatus


class DecisionParseError(ValueError):
    def __init__(self, reason: str, raw: str):
        super().__init__(reason)
        self.reason = reason
        self.raw = raw


def _balanced_candidates(text: str):
    """Yield every balanced {...} substring, aware of quoted strings."""
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : i + 1]
                    break


def _try_load(candidate: str):
    for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
        try:
            obj = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_json_object(text: str) -> dict:
    """Return the first JSON object that looks like a decision (has 'action')."""
    for candidate in _balanced_candidates(text):
        obj = _try_load(candidate)
        if obj is not None and "action" in obj:
            return obj
    raise DecisionParseError("no JSON object with an 'action' field found", text)


def parse_decision(text: str) -> AgentDecision:
    obj = extract_json_object(text)
    try:
        return AgentDecision.model_validate(obj)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'decision'}: {e['msg']}"
            for e in exc.errors()
        )
        raise DecisionParseError(f"schema check failed: {problems}", text) from exc


@dataclass
class DecisionResult:
    ok: bool
    decision: AgentDecision | None
    attempts: int
    status: str
    errors: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


def request_decision(model: ModelClient, messages: list[dict],
                     max_attempts: int = 3) -> DecisionResult:
    """Ask for a decision; on a rejected reply, explain why and retry (bounded)."""
    msgs = list(messages)
    result = DecisionResult(ok=False, decision=None, attempts=0,
                            status=TerminalStatus.FAILED_SAFELY.value)
    for _ in range(max_attempts):
        result.attempts += 1
        try:
            resp = model.generate(msgs)
        except ModelError as exc:
            result.errors.append(f"model error: {exc}")
            return result  # infrastructure failure: do not retry blindly
        result.prompt_tokens += resp.prompt_tokens or 0
        result.completion_tokens += resp.completion_tokens or 0
        result.latency_s += resp.latency_s
        try:
            result.decision = parse_decision(resp.text)
            result.ok = True
            result.status = "OK"
            return result
        except DecisionParseError as exc:
            result.errors.append(exc.reason)
            msgs.append({"role": "assistant", "content": resp.text})
            msgs.append({
                "role": "user",
                "content": (
                    f"Your last reply was rejected: {exc.reason}. "
                    "Reply again with ONE JSON object only, matching the schema."
                ),
            })
    return result
