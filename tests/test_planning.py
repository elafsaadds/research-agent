import pytest

from research_agent.evidence import EvidenceTable
from research_agent.planning import (PlanState, estimate_tool_calls, fallback_plan,
                                     overall_status, validate_plan)
from research_agent.policy import Budget
from research_agent.schemas import Finding, ResearchPlan, TerminalStatus

SRC = {"doc-a": "Indirect prompt injection hides instructions in web pages that an agent reads."}


def plan(n=2):
    return ResearchPlan(goal="g", sub_questions=[
        {"id": f"s{i}", "question": f"Sub question number {i} about agents"}
        for i in range(1, n + 1)])


def table_with(*sub_ids):
    t = EvidenceTable()
    for sid in sub_ids:
        t.add(sid, Finding(source_id="doc-a", claim="c",
                           evidence="hides instructions in web pages", stance="supports"), SRC)
    return t


def test_valid_plan():
    assert validate_plan(plan(2), Budget()) is None


def test_too_many_sub_questions():
    assert "too many" in validate_plan(plan(5), Budget())


def test_short_sub_question():
    p = ResearchPlan(goal="g", sub_questions=[{"id": "s1", "question": "short"}])
    assert "too short" in validate_plan(p, Budget())


def test_duplicate_sub_questions():
    p = ResearchPlan(goal="g", sub_questions=[
        {"id": "s1", "question": "What is indirect injection?"},
        {"id": "s2", "question": "what is   INDIRECT injection?"}])
    assert "duplicates" in validate_plan(p, Budget())


def test_plan_must_fit_the_tool_budget():
    assert "budget" in validate_plan(plan(2), Budget(max_tool_calls=3))


def test_estimate_tool_calls():
    assert estimate_tool_calls(plan(3), 2) == 6


def test_fallback_plan_is_valid():
    p = fallback_plan("What is excessive agency in agents?")
    assert [s.id for s in p.sub_questions] == ["s1"]
    assert validate_plan(p, Budget()) is None


def test_progress_state():
    s = PlanState(plan(2))
    assert [x.id for x in s.pending()] == ["s1", "s2"]
    s.mark("s1", "done")
    assert [x.id for x in s.pending()] == ["s2"]


def test_replan_accepted_resets_status():
    s = PlanState(plan(2))
    s.mark("s1", "failed")
    assert s.try_replan("s1", "A different wording for this sub question") is None
    assert s.get("s1").status == "pending" and s.replans == 1
    assert s.get("s1").question.startswith("A different wording")


def test_replan_refuses_repeating_the_original():
    s = PlanState(plan(2))
    reason = s.try_replan("s1", "  SUB QUESTION number 1 about AGENTS ")
    assert "repeats" in reason and s.replans == 0


def test_replan_budget_is_bounded():
    s = PlanState(plan(2), max_replans=1)
    assert s.try_replan("s1", "First different wording here") is None
    assert "exhausted" in s.try_replan("s2", "Second different wording here")


def test_replan_refuses_too_short_replacement():
    assert "too short" in PlanState(plan(2)).try_replan("s1", "hi")


def test_overall_status_none_answered():
    assert overall_status(PlanState(plan(2)), EvidenceTable()) == TerminalStatus.INSUFFICIENT_EVIDENCE


def test_overall_status_all_answered():
    assert overall_status(PlanState(plan(2)), table_with("s1", "s2")) == TerminalStatus.COMPLETED


def test_overall_status_partial():
    assert overall_status(PlanState(plan(2)), table_with("s1")) == TerminalStatus.PARTIAL
