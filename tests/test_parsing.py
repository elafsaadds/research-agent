import json

import pytest

from research_agent.model import ModelResponse
from research_agent.parsing import (DecisionParseError, parse_decision,
                                    request_decision)

FENCE = chr(96) * 3
GOOD = {"action": "tool", "tool_name": "search_corpus",
        "arguments": {"query": "prompt injection"},
        "reason_summary": "need background"}


class ScriptedModel:
    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.seen = []

    def generate(self, messages):
        self.seen.append(list(messages))
        text = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return ModelResponse(text=text, model="scripted", latency_s=0.0,
                             prompt_tokens=1, completion_tokens=1)


def test_clean_json():
    assert parse_decision(json.dumps(GOOD)).tool_name == "search_corpus"


def test_fenced_json():
    text = f"{FENCE}json\n{json.dumps(GOOD)}\n{FENCE}"
    assert parse_decision(text).action == "tool"


def test_prose_around_json():
    text = "Sure! Here is my decision: " + json.dumps(GOOD) + " Hope it helps."
    assert parse_decision(text).action == "tool"


def test_trailing_comma_is_repaired():
    text = '{"action": "tool", "tool_name": "search_corpus", "reason_summary": "r",}'
    assert parse_decision(text).tool_name == "search_corpus"


def test_braces_inside_string_are_safe():
    d = {"action": "answer", "message": "use {x} carefully", "reason_summary": "r"}
    assert parse_decision(json.dumps(d)).message == "use {x} carefully"


def test_no_json_raises():
    with pytest.raises(DecisionParseError):
        parse_decision("I think you should search for it.")


def test_wrong_schema_raises():
    bad = json.dumps({"action": "tool", "reason_summary": "no tool name"})
    with pytest.raises(DecisionParseError) as exc:
        parse_decision(bad)
    assert "tool_name" in exc.value.reason


def test_retry_then_success():
    model = ScriptedModel(["not json at all", json.dumps(GOOD)])
    r = request_decision(model, [{"role": "user", "content": "q"}])
    assert r.ok and r.attempts == 2 and len(r.errors) == 1
    assert "rejected" in model.seen[1][-1]["content"]


def test_gives_up_after_max_attempts():
    model = ScriptedModel(["nope"])
    r = request_decision(model, [{"role": "user", "content": "q"}], max_attempts=3)
    assert not r.ok and r.attempts == 3 and r.status == "FAILED_SAFELY"
