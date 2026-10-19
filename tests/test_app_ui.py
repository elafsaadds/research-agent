from app import ui


def test_show_json_none():
    assert ui.show_json(None) == {}


def test_show_json_dictionary():
    value = {"status": "COMPLETED", "count": 3}

    assert ui.show_json(value) == value


def test_show_json_json_string():
    value = '{"status": "COMPLETED", "count": 3}'

    assert ui.show_json(value) == {
        "status": "COMPLETED",
        "count": 3,
    }


def test_show_json_invalid_string():
    value = "not valid json"

    assert ui.show_json(value) == {
        "value": "not valid json",
    }


def test_show_json_list():
    value = ["one", "two", "three"]

    assert ui.show_json(value) == value


def test_as_dict_accepts_dictionary():
    value = {"status": "COMPLETED"}

    assert ui.as_dict(value) == value


def test_as_dict_rejects_non_dictionary():
    assert ui.as_dict("invalid") == {}
    assert ui.as_dict(None) == {}
    assert ui.as_dict([]) == {}


def test_research_callback_returns_all_outputs(monkeypatch):
    response = {
        "question": "What is AI?",
        "model": "fake:test-model",
        "status": "COMPLETED",
        "message": "Research completed.",
        "result": {
            "plan": {
                "goal": "What is AI?",
            },
            "table": {
                "rows": [],
            },
            "conflicts": [],
            "answered": 1,
            "total": 1,
            "replans": 0,
            "original_answered": True,
            "model_calls": 2,
            "findings_rejected": 0,
            "latency_s": 0.5,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "search_log": [],
        },
    }

    monkeypatch.setattr(
        ui,
        "run_research_request",
        lambda question: response,
    )

    report, plan, evidence, conflicts, diagnostics = (
        ui.research_callback("What is AI?")
    )

    assert "# Research Report" in report
    assert "**Status:** COMPLETED" in report
    assert "**Model:** fake:test-model" in report
    assert "Research completed." in report

    assert plan == {
        "goal": "What is AI?",
    }

    assert evidence == {
        "rows": [],
    }

    assert conflicts == []

    assert diagnostics["question"] == "What is AI?"
    assert diagnostics["status"] == "COMPLETED"
    assert diagnostics["model"] == "fake:test-model"
    assert diagnostics["answered"] == 1
    assert diagnostics["total_sub_questions"] == 1
    assert diagnostics["original_question_answered"] is True


def test_research_callback_handles_pipeline_error(monkeypatch):
    def fail(question):
        raise RuntimeError("pipeline failed")

    monkeypatch.setattr(
        ui,
        "run_research_request",
        fail,
    )

    report, plan, evidence, conflicts, diagnostics = (
        ui.research_callback("What is AI?")
    )

    assert "# Research run failed" in report
    assert "RuntimeError" in report
    assert "pipeline failed" in report

    expected_error = {
        "error_type": "RuntimeError",
        "error": "pipeline failed",
    }

    assert plan == expected_error
    assert evidence == expected_error
    assert conflicts == expected_error
    assert diagnostics == expected_error