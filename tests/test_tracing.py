import json
from datetime import datetime, timezone

import pytest

from research_agent import agent, tracing
from research_agent.model import ModelResponse
from research_agent.policy import Budget
from research_agent.tracing import (RUN_ID_RE, TraceStore, action_sequence,
                                    compare_runs, corpus_fingerprint,
                                    format_timeline, new_run_id, redact,
                                    redact_text, traced_run)
from research_agent.tools import SearchHit, ToolResult


def j(**kw):
    return json.dumps(kw)


def tool_reply(q):
    return j(action="tool", tool_name="search_corpus",
             arguments={"query": q}, reason_summary="r")


def answer_reply(msg):
    return j(action="answer", message=msg, reason_summary="r")


def insufficient_reply():
    return j(action="insufficient_evidence", message="nothing found", reason_summary="r")


class ScriptedModel:
    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def generate(self, messages):
        text = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return ModelResponse(text=text, model="scripted", latency_s=0.0,
                             prompt_tokens=5, completion_tokens=3)


HIT = SearchHit(source_id="doc-001", title="Injection",
                snippet="Hidden instructions in pages.", origin="corpus", score=1.0)


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    monkeypatch.setattr(agent, "execute",
                        lambda name, arguments: ToolResult(tool=name, ok=True, hits=[HIT]))
    monkeypatch.setattr(tracing, "_git_info", lambda: {"commit": "abc1234", "dirty": False})


@pytest.fixture
def store(tmp_path):
    return TraceStore(tmp_path / "runs")


def run(store, replies, **kw):
    return traced_run("What is indirect prompt injection?", ScriptedModel(replies),
                      Budget(), store, **kw)


def test_run_id_format_and_uniqueness():
    ids = {new_run_id(datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)) for _ in range(30)}
    assert all(RUN_ID_RE.match(i) for i in ids)
    assert len(ids) > 1


def test_record_is_saved_and_loads_back(store):
    rec = run(store, [tool_reply("q one"), answer_reply("claim [doc-001]")])
    loaded = store.load(rec.run_id)
    assert loaded == rec
    assert loaded.status == "COMPLETED"
    assert loaded.counters["tool_calls"] == 1
    assert loaded.tokens["prompt"] > 0 and loaded.latency_s >= 0


def test_record_names_its_versions(store):
    rec = run(store, [tool_reply("q one"), answer_reply("claim [doc-001]")])
    v = rec.versions
    assert v["model"] == "scripted" and v["tools"] == tracing.TOOLS_VERSION
    assert v["code"] == "abc1234" and v["retrieval"].startswith("bm25:")
    assert "+registered-tools" in v["prompt"]


def test_budget_is_recorded_and_serialisable(store):
    rec = run(store, [tool_reply("q one"), answer_reply("claim [doc-001]")])
    assert rec.budget["max_turns"] == 8 and rec.budget["allowed_tiers"] == ["read"]


def test_trace_keeps_ids_not_snippets(store):
    rec = run(store, [tool_reply("q one"), answer_reply("claim [doc-001]")])
    text = rec.model_dump_json()
    assert "doc-001" in text
    assert "Hidden instructions in pages" not in text


def test_failed_run_is_still_recorded(store):
    rec = run(store, ["not json"])
    assert rec.status == "FAILED_SAFELY"
    assert any(e["event"] == "model_failure" for e in rec.events)
    assert store.load(rec.run_id).status == "FAILED_SAFELY"


def test_no_temp_files_left_after_save(store):
    run(store, [tool_reply("q one"), answer_reply("claim [doc-001]")])
    assert list(store.dir.glob("*.tmp")) == []


def test_invalid_run_id_is_refused(store):
    with pytest.raises(ValueError):
        store.load("../../etc/passwd")
    with pytest.raises(ValueError):
        store.load("run_1")


def test_missing_run_raises(store):
    with pytest.raises(FileNotFoundError):
        store.load("run_20260101_000000_aaaa")


def test_list_is_newest_first_and_skips_corrupt_files(store):
    rec = run(store, [tool_reply("q one"), answer_reply("claim [doc-001]")])
    older = rec.model_copy(update={"run_id": "run_20260101_000000_aaaa"})
    newer = rec.model_copy(update={"run_id": "run_20270101_000000_bbbb"})
    store.save(older)
    store.save(newer)
    (store.dir / "run_20260101_000001_cccc.json").write_text("{", encoding="utf-8")
    ids = [r["run_id"] for r in store.list()]
    assert ids[0] == "run_20270101_000000_bbbb"
    assert "run_20260101_000001_cccc" not in ids
    assert len(store.list(limit=1)) == 1


def test_empty_store_lists_nothing(tmp_path):
    assert TraceStore(tmp_path / "nothing").list() == []


def test_redact_secret_patterns():
    out = redact_text("key sk-abcdefghijklmnopqrstuvwxyz123456 and Bearer abcdefghijklmnop1234")
    assert "sk-abcdefghij" not in out and "abcdefghijklmnop1234" not in out
    assert out.count("[REDACTED]") == 2


def test_redact_env_secret_values(monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "supersecretvalue123")
    out = redact({"a": ["token supersecretvalue123 here"]})
    assert "supersecretvalue123" not in json.dumps(out)


def test_redact_truncates_long_text():
    out = redact_text("x" * 5000)
    assert len(out) < 2100 and "truncated" in out


def test_secret_in_question_is_not_stored(store, monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "supersecretvalue123")
    rec = traced_run("my key is supersecretvalue123", ScriptedModel([insufficient_reply()]),
                     Budget(), store)
    assert "supersecretvalue123" not in store.load(rec.run_id).model_dump_json()


def test_corpus_fingerprint_changes_with_content(tmp_path):
    assert corpus_fingerprint(tmp_path / "missing") == "none"
    (tmp_path / "a.md").write_text("one", encoding="utf-8")
    first = corpus_fingerprint(tmp_path)
    (tmp_path / "a.md").write_text("two", encoding="utf-8")
    assert corpus_fingerprint(tmp_path) != first


def test_fingerprint_ignores_line_endings(tmp_path):
    (tmp_path / "a.md").write_bytes(b"line\nline")
    unix = corpus_fingerprint(tmp_path)
    (tmp_path / "a.md").write_bytes(b"line\r\nline")
    assert corpus_fingerprint(tmp_path) == unix


def test_timeline_shows_rejection_and_final(store):
    rec = run(store, [insufficient_reply(), tool_reply("q one"), answer_reply("claim [doc-001]")])
    text = "\n".join(format_timeline(rec))
    assert "REJECTED [policy]" in text and "TOOL" in text and "FINAL   COMPLETED" in text


def test_action_sequence_marks_rejections(store):
    rec = run(store, [insufficient_reply(), tool_reply("q one"), answer_reply("claim [doc-001]")])
    seq = action_sequence(rec)
    assert seq[0].endswith("REJECTED[policy]")
    assert seq[1] == "tool|search_corpus|q one"


def test_compare_finds_version_and_behaviour_differences(store):
    a = run(store, [tool_reply("q one"), answer_reply("claim [doc-001]")], prompt_version="v2")
    b = run(store, [insufficient_reply(), tool_reply("q one"), answer_reply("claim [doc-001]")],
            prompt_version="v3")
    diff = compare_runs(a, b)
    assert diff["same_question"] is True
    assert diff["version_diffs"]["prompt"] == ("v2", "v3")
    assert diff["first_divergence"] == 0
    assert diff["counter_diffs"]["rejections"] == (0, 1)


def test_compare_identical_behaviour_has_no_divergence(store):
    a = run(store, [tool_reply("q one"), answer_reply("claim [doc-001]")])
    b = run(store, [tool_reply("q one"), answer_reply("claim [doc-001]")])
    assert compare_runs(a, b)["first_divergence"] is None
