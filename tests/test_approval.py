
import pytest

from research_agent.approval import run_approval_demo


def test_pending_approval_does_not_execute():
    result = run_approval_demo("pending")

    assert result.status == "WAITING_FOR_APPROVAL"
    assert result.executed is False
    assert result.events[-1]["event"] == "approval_pending"


def test_rejected_action_does_not_execute():
    result = run_approval_demo("rejected")

    assert result.status == "BLOCKED"
    assert result.executed is False
    assert result.events[-1]["event"] == "protected_action_blocked"


def test_approved_action_is_simulated():
    result = run_approval_demo("approved")

    assert result.status == "COMPLETED"
    assert result.executed is True
    assert result.events[-1]["event"] == "protected_action_simulated"


def test_invalid_approval_decision_is_rejected():
    with pytest.raises(ValueError):
        run_approval_demo("maybe")
