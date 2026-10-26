
"""
Validation gates: the code that can say NO to the model.

Order:
1. Schema validation is handled by AgentDecision.
2. Reference and argument validation.
3. Authority validation.
4. Policy validation.

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


# Matches citations such as:
# [arxiv:2304.04780v1]
# [openalex:W7207768306]
# [corpus:document_name]
CITATION = re.compile(r"\[([A-Za-z0-9:._\-]+)\]")


# Required search sources for the current tested workflow.
# OpenAlex remains registered as a tool, but is not yet required here.
REQUIRED_SEARCH_TOOLS = frozenset(
    {
        "search_corpus",
        "search_arxiv",
    }
)


@dataclass(frozen=True)
class Budget:
    max_turns: int = 8
    max_tool_calls: int = 18
    run_timeout_s: float = 60.0

    # Maximum consecutive rejected proposals.
    max_rejections: int = 3

    # Approximate context limit for the agent.
    max_context_chars: int = 12000

    # The agent is allowed to use read-only tools.
    allowed_tiers: frozenset = frozenset({"read"})

    @classmethod
    def from_env(cls) -> "Budget":
        load_dotenv()

        return cls(
            max_turns=int(
                os.getenv("MAX_AGENT_TURNS", "8")
            ),
            max_tool_calls=int(
                os.getenv("MAX_TOOL_CALLS", "18")
            ),
            run_timeout_s=float(
                os.getenv("RUN_TIMEOUT_SECONDS", "60")
            ),
        )


@dataclass
class RunState:
    turns: int = 0
    tool_calls: int = 0
    searches_ok: int = 0
    rejections: int = 0
    consecutive_rejections: int = 0

    calls: set[str] = field(default_factory=set)

    known_sources: dict[str, SearchHit] = field(
        default_factory=dict
    )

    tried_tools: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    gate: str = "passed"
    reason: str = ""

    def feedback(self) -> str:
        return (
            f"Your proposed action was rejected by the {self.gate} gate: "
            f"{self.reason} Propose a different action. "
            "Reply with ONE JSON object only."
        )


def _reject(
    gate: str,
    reason: str,
) -> GateResult:
    """Return a structured rejection from a validation gate."""
    return GateResult(
        allowed=False,
        gate=gate,
        reason=reason,
    )


def call_key(
    name: str,
    arguments: dict | None,
    registry: dict[str, ToolSpec] | None = None,
) -> str:
    """
    Create a canonical key for a tool call.

    This prevents the agent from repeating the same call by changing
    only insignificant whitespace or letter casing in string arguments.
    """

    registry = REGISTRY if registry is None else registry
    args = dict(arguments or {})

    spec = registry.get(name)

    if spec is not None:
        try:
            args = (
                spec.args_model
                .model_validate(args)
                .model_dump()
            )
        except ValidationError:
            # Keep the original arguments if validation fails.
            # The reference gate will reject invalid arguments later.
            pass

    norm = {
        key: (
            " ".join(value.casefold().split())
            if isinstance(value, str)
            else value
        )
        for key, value in sorted(args.items())
    }

    return (
        name
        + ":"
        + json.dumps(
            norm,
            sort_keys=True,
            default=str,
        )
    )


def validate_decision(
    d: AgentDecision,
    state: RunState,
    budget: Budget,
    registry: dict[str, ToolSpec] | None = None,
) -> GateResult:
    """
    Validate an agent decision before the action is executed.

    Tool decisions pass through reference, argument, authority,
    and repeat-call checks. Other decisions pass through policy checks.
    """

    registry = REGISTRY if registry is None else registry

    # -------------------------------------------------------------
    # 1. Tool reference, argument, authority, and repeat-call checks.
    # -------------------------------------------------------------

    if d.action == "tool":
        tool_name = d.tool_name or ""
        spec = registry.get(tool_name)

        # Reference gate: the tool must exist.
        if spec is None:
            return _reject(
                "reference",
                f"Unknown tool {tool_name!r}. Available tools: "
                f"{', '.join(sorted(registry))}.",
            )

        # Reference gate: arguments must match the tool schema.
        try:
            spec.args_model.model_validate(
                d.arguments or {}
            )
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'arguments'}: "
                f"{error['msg']}"
                for error in exc.errors()
            )

            return _reject(
                "reference",
                f"Invalid arguments: {problems}.",
            )

        # Authority gate: only permitted tool tiers may be used.
        if spec.tier not in budget.allowed_tiers:
            return _reject(
                "authority",
                f"Tool {spec.name!r} has tier {spec.tier!r}, "
                "which this agent may not use.",
            )

        # Policy gate: do not repeat an identical tool call.
        key = call_key(
            spec.name,
            d.arguments,
            registry,
        )

        if key in state.calls:
            return _reject(
                "policy",
                "This exact call was already made. "
                "Change the query or provide an answer.",
            )

        return GateResult(allowed=True)

    # -------------------------------------------------------------
    # 2. The agent cannot give up before trying required sources.
    # -------------------------------------------------------------

    if (
        d.action == "insufficient_evidence"
        and state.searches_ok > 0
    ):
        untried = sorted(
            name
            for name in REQUIRED_SEARCH_TOOLS
            if (
                name in registry
                and name not in state.tried_tools
            )
        )

        if untried:
            return _reject(
                "policy",
                "insufficient_evidence is not allowed until every "
                "required search source has been tried. "
                "Not yet tried: "
                + ", ".join(untried)
                + ".",
            )

    # -------------------------------------------------------------
    # 3. Asking the user or declaring insufficient evidence requires
    #    at least one successful search.
    # -------------------------------------------------------------

    if d.action in (
        "insufficient_evidence",
        "ask_user",
    ):
        if state.searches_ok == 0:
            return _reject(
                "policy",
                f"{d.action} is not allowed before a successful search. "
                "Search first.",
            )

        return GateResult(allowed=True)

    # -------------------------------------------------------------
    # 4. Answers must cite evidence retrieved during this run.
    # -------------------------------------------------------------

    if d.action == "answer":
        if not state.known_sources:
            return _reject(
                "policy",
                "No evidence has been retrieved yet. Search first, "
                "or use insufficient_evidence if searches found nothing.",
            )

        cited = list(
            dict.fromkeys(
                CITATION.findall(d.message or "")
            )
        )

        if not cited:
            return _reject(
                "policy",
                "The answer must cite retrieved sources using "
                "[source_id]. Retrieved IDs: "
                + ", ".join(sorted(state.known_sources))
                + ".",
            )

        unknown = [
            citation
            for citation in cited
            if citation not in state.known_sources
        ]

        if unknown:
            return _reject(
                "policy",
                "The answer cites sources that were never retrieved: "
                + ", ".join(unknown)
                + ". Retrieved IDs: "
                + ", ".join(sorted(state.known_sources))
                + ".",
            )

        return GateResult(allowed=True)

    # -------------------------------------------------------------
    # 5. Unknown or unsupported actions are rejected.
    # -------------------------------------------------------------

    return _reject(
        "schema",
        f"Unknown action {d.action!r}.",
    )