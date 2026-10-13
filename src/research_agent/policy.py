"""Validation gates: the code that can say NO to the model.

Order: schema (AgentDecision) -> reference -> authority -> policy.
A rejection is data: the loop records it and tells the model why.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic import ValidationError

from research_agent.schemas import AgentDecision
from research_agent.tools import REGISTRY, SearchHit, ToolSpec

CITATION = re.compile(r"\[([A-Za-z0-9:._\-]+)\]")


@dataclass(frozen=True)
class Budget:
    max_turns: int = 8
    max_tool_calls: int = 12
    run_timeout_s: float = 60.0
    max_rejections: int = 3          # consecutive rejected proposals
    max_context_chars: int = 12000   # ~3000 tokens: stay inside num_ctx 4096
    allowed_tiers: frozenset = frozenset({"read"})

    @classmethod
    def from_env(cls) -> "Budget":
        load_dotenv()
        return cls(
            max_turns=int(os.getenv("MAX_AGENT_TURNS", "8")),
            max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "12")),
            run_timeout_s=float(os.getenv("RUN_TIMEOUT_SECONDS", "60")),
        )


@dataclass
class RunState:
    turns: int = 0
    tool_calls: int = 0
    searches_ok: int = 0
    rejections: int = 0
    consecutive_rejections: int = 0
    calls: set[str] = field(default_factory=set)
    known_sources: dict[str, SearchHit] = field(default_factory=dict)
    tried_tools: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    gate: str = "passed"
    reason: str = ""

    def feedback(self) -> str:
        return (f"Your proposed action was rejected by the {self.gate} gate: "
                f"{self.reason} Propose a different action. "
                "Reply with ONE JSON object only.")


def _reject(gate: str, reason: str) -> GateResult:
    return GateResult(False, gate, reason)


def call_key(name: str, arguments: dict | None,
             registry: dict[str, ToolSpec] | None = None) -> str:
    """Canonical form of a call, so trivial rewording does not evade the repeat check."""
    registry = REGISTRY if registry is None else registry
    args = dict(arguments or {})
    spec = registry.get(name)
    if spec is not None:
        try:
            args = spec.args_model.model_validate(args).model_dump()
        except ValidationError:
            pass
    norm = {k: (" ".join(v.casefold().split()) if isinstance(v, str) else v)
            for k, v in sorted(args.items())}
    return name + ":" + json.dumps(norm, sort_keys=True, default=str)


def validate_decision(d: AgentDecision, state: RunState, budget: Budget,
                      registry: dict[str, ToolSpec] | None = None) -> GateResult:
    registry = REGISTRY if registry is None else registry

    if d.action == "tool":
        spec = registry.get(d.tool_name or "")
        if spec is None:
            return _reject("reference",
                           f"unknown tool {d.tool_name!r}. Available tools: "
                           f"{', '.join(sorted(registry))}.")
        try:
            spec.args_model.model_validate(d.arguments or {})
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
                for e in exc.errors())
            return _reject("reference", f"invalid arguments: {problems}.")
        if spec.tier not in budget.allowed_tiers:
            return _reject("authority",
                           f"tool {spec.name!r} has tier {spec.tier!r}, which this agent may not use.")
        if call_key(spec.name, d.arguments, registry) in state.calls:
            return _reject("policy",
                           "this exact call was already made. Change the query or answer.")
        return GateResult(True)

    if d.action == "insufficient_evidence" and state.searches_ok > 0:
        untried = sorted(n for n in registry
                         if n.startswith("search_") and n not in state.tried_tools)
        if untried:
            return _reject("policy",
                           "insufficient_evidence is not allowed until every search source "
                           "has been tried. Not yet tried: " + ", ".join(untried) + ".")
    if d.action in ("insufficient_evidence", "ask_user"):
        if state.searches_ok == 0:
            return _reject("policy",
                           f"{d.action} is not allowed before a successful search. Search first.")
        return GateResult(True)

    if d.action == "answer":
        if not state.known_sources:
            return _reject("policy",
                           "no evidence has been retrieved yet. Search first, or use "
                           "insufficient_evidence if searches found nothing.")
        cited = list(dict.fromkeys(CITATION.findall(d.message or "")))
        if not cited:
            return _reject("policy",
                           "the answer must cite retrieved sources as [source_id]. Retrieved ids: "
                           + ", ".join(sorted(state.known_sources)) + ".")
        unknown = [c for c in cited if c not in state.known_sources]
        if unknown:
            return _reject("policy",
                           f"the answer cites sources that were never retrieved: "
                           f"{', '.join(unknown)}. Retrieved ids: "
                           f"{', '.join(sorted(state.known_sources))}.")
        return GateResult(True)

    return _reject("schema", f"unknown action {d.action!r}.")

