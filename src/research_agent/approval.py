"""Human approval gate for a protected action.

This module does not perform any external or destructive action.
The protected action is only simulated, and only when a human
decision of "approved" is supplied.
"""
from dataclasses import dataclass
from typing import Literal


ApprovalDecision = Literal["approved", "rejected", "pending"]

PROTECTED_ACTION = "simulate_publish_research_report"


@dataclass(frozen=True)
class ApprovalResult:
    status: str
    action: str
    executed: bool
    events: list[dict]


def run_approval_demo(
    decision: ApprovalDecision = "pending",
) -> ApprovalResult:
    """Demonstrate that a protected action requires explicit approval."""
    return request_approval(PROTECTED_ACTION, decision)


def request_approval(
    action: str,
    decision: ApprovalDecision = "pending",
) -> ApprovalResult:
    """Gate a protected action behind an explicit human decision.

    The action is executed only when the decision is "approved".
    Any other value (including an unknown one) never executes.
    """
    if decision not in {"approved", "rejected", "pending"}:
        raise ValueError(
            "decision must be approved, rejected, or pending"
        )

    events = [
        {
            "event": "approval_requested",
            "action": action,
            "message": "Human approval is required before proceeding.",
        }
    ]

    if decision == "pending":
        events.append({
            "event": "approval_pending",
            "action": action,
        })
        return ApprovalResult(
            status="WAITING_FOR_APPROVAL",
            action=action,
            executed=False,
            events=events,
        )

    events.append({
        "event": "approval_decision",
        "action": action,
        "decision": decision,
    })

    if decision == "rejected":
        events.append({
            "event": "protected_action_blocked",
            "action": action,
        })
        return ApprovalResult(
            status="BLOCKED",
            action=action,
            executed=False,
            events=events,
        )

    # Simulation only: no real publishing operation occurs.
    events.append({
        "event": "protected_action_simulated",
        "action": action,
    })
    return ApprovalResult(
        status="COMPLETED",
        action=action,
        executed=True,
        events=events,
    )
