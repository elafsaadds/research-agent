import pytest

from research_agent.policy import Budget, RunState, call_key, validate_decision
from research_agent.schemas import AgentDecision
from research_agent.tools import QueryArgs, SearchHit, ToolSpec

BUDGET = Budget()


def tool(name="search_corpus", **args):
    return AgentDecision(action="tool", tool_name=name,
                         arguments=args or {"query": "prompt injection"},
                         reason_summary="r")


def other(action, message="text [doc-001]"):
    return AgentDecision(action=action, message=message, reason_summary="r")


def hit(i="doc-001"):
    return SearchHit(source_id=i, title="t", snippet="s", origin="corpus")


def with_evidence():
    s = RunState()
    s.searches_ok = 1
    s.tried_tools = {"search_corpus", "search_arxiv"}
    s.known_sources["doc-001"] = hit()
    return s


def test_valid_search_is_allowed():
    assert validate_decision(tool(), RunState(), BUDGET).allowed


def test_unknown_tool_rejected():
    r = validate_decision(tool("delete_everything"), RunState(), BUDGET)
    assert not r.allowed and r.gate == "reference"


def test_invalid_arguments_rejected():
    r = validate_decision(tool(query="x"), RunState(), BUDGET)
    assert not r.allowed and r.gate == "reference"


def test_extra_argument_rejected():
    r = validate_decision(tool(query="valid query", secret="1"), RunState(), BUDGET)
    assert not r.allowed and r.gate == "reference"


def test_write_tier_blocked_by_authority():
    spec = ToolSpec("save_report", "write", "d", QueryArgs, lambda a: [], 1.0, "file")
    r = validate_decision(tool("save_report", query="valid text"), RunState(), BUDGET,
                          registry={"save_report": spec})
    assert not r.allowed and r.gate == "authority"


def test_repeated_call_rejected_even_when_reworded():
    s = RunState()
    s.calls.add(call_key("search_corpus", {"query": "Prompt   Injection"}))
    r = validate_decision(tool(query="prompt injection"), s, BUDGET)
    assert not r.allowed and r.gate == "policy"


def test_different_query_is_not_a_repeat():
    s = RunState()
    s.calls.add(call_key("search_corpus", {"query": "prompt injection"}))
    assert validate_decision(tool(query="memory poisoning"), s, BUDGET).allowed


def test_insufficient_evidence_before_search_rejected():
    r = validate_decision(other("insufficient_evidence", "nothing"), RunState(), BUDGET)
    assert not r.allowed and r.gate == "policy"


def test_insufficient_evidence_after_search_allowed():
    assert validate_decision(other("insufficient_evidence", "nothing"),
                             with_evidence(), BUDGET).allowed


def test_failed_search_does_not_unlock_insufficient_evidence():
    s = RunState()
    s.tool_calls = 1   # a call happened but it failed: searches_ok stays 0
    assert not validate_decision(other("insufficient_evidence", "nothing"), s, BUDGET).allowed


def test_ask_user_gated_like_insufficient_evidence():
    assert not validate_decision(other("ask_user", "which one?"), RunState(), BUDGET).allowed
    assert validate_decision(other("ask_user", "which one?"), with_evidence(), BUDGET).allowed


def test_answer_without_evidence_rejected():
    r = validate_decision(other("answer"), RunState(), BUDGET)
    assert not r.allowed and r.gate == "policy"


def test_answer_without_citation_rejected():
    r = validate_decision(other("answer", "no citation here"), with_evidence(), BUDGET)
    assert not r.allowed


def test_answer_citing_unretrieved_source_rejected():
    r = validate_decision(other("answer", "claim [doc-999]"), with_evidence(), BUDGET)
    assert not r.allowed and "doc-999" in r.reason


def test_answer_with_valid_citation_allowed():
    assert validate_decision(other("answer", "claim [doc-001]"), with_evidence(), BUDGET).allowed


def test_feedback_names_the_gate():
    r = validate_decision(tool("nope"), RunState(), BUDGET)
    assert "reference" in r.feedback()


def test_budget_from_env(monkeypatch):
    monkeypatch.setenv("MAX_AGENT_TURNS", "3")
    monkeypatch.setenv("RUN_TIMEOUT_SECONDS", "12.5")
    b = Budget.from_env()
    assert b.max_turns == 3 and b.run_timeout_s == 12.5


def test_insufficient_evidence_needs_every_search_source_tried():
    s = with_evidence()
    s.tried_tools = {"search_corpus"}
    r = validate_decision(other("insufficient_evidence", "nothing"), s, BUDGET)
    assert not r.allowed and r.gate == "policy" and "search_arxiv" in r.reason


def test_no_citation_feedback_lists_retrieved_ids():
    r = validate_decision(other("answer", "no citation here"), with_evidence(), BUDGET)
    assert not r.allowed and "doc-001" in r.reason
