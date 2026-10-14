"""Typed structures: what the model may propose, and how a run may end."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TerminalStatus(str, Enum):
    """Every run must end in exactly one of these."""

    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    BLOCKED = "BLOCKED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    FAILED_SAFELY = "FAILED_SAFELY"


class AgentDecision(BaseModel):
    """One proposal from the model. The application decides whether to act on it."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["tool", "answer", "ask_user", "insufficient_evidence"]
    reason_summary: str = Field(min_length=1, max_length=500)
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    message: str | None = None  # final report, question to user, or explanation

    @model_validator(mode="after")
    def check_shape(self) -> "AgentDecision":
        if self.action == "tool":
            if not self.tool_name:
                raise ValueError("tool_name is required when action is 'tool'")
            if self.arguments is None:
                self.arguments = {}
        else:
            if self.tool_name is not None or self.arguments:
                raise ValueError(
                    "tool_name and arguments are only allowed when action is 'tool'"
                )
            if not (self.message and self.message.strip()):
                raise ValueError(
                    f"message is required when action is '{self.action}'"
                )
        return self


class Finding(BaseModel):
    """One piece of evidence. Used by the evidence table in stage 7."""

    source_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    stance: Literal["supports", "contradicts", "uncertain"] = "supports"


class SubQuestion(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    status: Literal["pending", "done", "failed"] = "pending"


class ResearchPlan(BaseModel):
    """Explicit plan with progress state (used from stage 7)."""

    goal: str = Field(min_length=1)
    sub_questions: list[SubQuestion] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def unique_ids(self) -> "ResearchPlan":
        ids = [s.id for s in self.sub_questions]
        if len(ids) != len(set(ids)):
            raise ValueError("sub-question ids must be unique")
        return self


def decision_json_schema() -> dict:
    """Schema for constrained output. Two variants make invalid mixes impossible."""
    reason = {"type": "string"}
    return {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["tool"]},
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object"},
                    "reason_summary": reason,
                },
                "required": ["action", "tool_name", "arguments", "reason_summary"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["answer", "ask_user", "insufficient_evidence"],
                    },
                    "message": {"type": "string"},
                    "reason_summary": reason,
                },
                "required": ["action", "message", "reason_summary"],
                "additionalProperties": False,
            },
        ]
    }
