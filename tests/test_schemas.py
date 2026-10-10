import pytest
from pydantic import ValidationError

from research_agent.schemas import AgentDecision, ResearchPlan


def test_valid_tool_decision():
    d = AgentDecision(action="tool", tool_name="search_corpus",
                      arguments={"query": "x"}, reason_summary="need evidence")
    assert d.tool_name == "search_corpus"


def test_tool_requires_name():
    with pytest.raises(ValidationError):
        AgentDecision(action="tool", reason_summary="oops")


def test_answer_requires_message():
    with pytest.raises(ValidationError):
        AgentDecision(action="answer", reason_summary="done")


def test_non_tool_rejects_tool_name():
    with pytest.raises(ValidationError):
        AgentDecision(action="answer", message="ok", tool_name="search_corpus",
                      reason_summary="done")


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        AgentDecision(action="answer", message="ok", reason_summary="r",
                      secret_field="x")


def test_plan_ids_must_be_unique():
    with pytest.raises(ValidationError):
        ResearchPlan(goal="g", sub_questions=[
            {"id": "s1", "question": "a"}, {"id": "s1", "question": "b"}])
