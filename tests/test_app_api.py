import pytest

from app import api


class FakeModel:
    name = "fake:test-model"


class FakeResult:
    status = "COMPLETED"
    message = "Research completed."
    model = "fake:test-model"

    def __init__(self):
        self.answer = "Test answer"
        self.sources = ["doc-001"]


def test_validate_question_accepts_valid_question():
    result = api.validate_question("  What is artificial intelligence?  ")

    assert result == "What is artificial intelligence?"


def test_validate_question_rejects_empty_question():
    with pytest.raises(ValueError, match="Please enter a research question"):
        api.validate_question("   ")


def test_validate_question_rejects_non_string():
    with pytest.raises(
        ValueError,
        match="Research question must be text",
    ):
        api.validate_question(None)


def test_validate_question_rejects_question_over_maximum_length():
    question = "x" * (api.MAX_QUESTION_LENGTH + 1)

    with pytest.raises(
        ValueError,
        match="Maximum length is 2000 characters",
    ):
        api.validate_question(question)


def test_validate_question_accepts_maximum_length():
    question = "x" * api.MAX_QUESTION_LENGTH

    assert api.validate_question(question) == question


def test_to_plain_data_converts_enum():
    from enum import Enum

    class Status(Enum):
        COMPLETED = "COMPLETED"

    assert api.to_plain_data(Status.COMPLETED) == "COMPLETED"


def test_to_plain_data_converts_dataclass():
    from dataclasses import dataclass

    @dataclass
    class Sample:
        name: str
        value: int

    result = api.to_plain_data(
        Sample(name="test", value=123)
    )

    assert result == {
        "name": "test",
        "value": 123,
    }


def test_to_plain_data_converts_nested_structures():
    value = {
        "items": (
            "one",
            "two",
        ),
        "nested": {
            "value": 10,
        },
    }

    result = api.to_plain_data(value)

    assert result == {
        "items": [
            "one",
            "two",
        ],
        "nested": {
            "value": 10,
        },
    }

def test_build_response_returns_expected_structure():
    model = FakeModel()
    result = FakeResult()

    response = api.build_response(
        question="What is AI?",
        result=result,
        model=model,
    )

    assert response["question"] == "What is AI?"
    assert response["model"] == "fake:test-model"
    assert response["status"] == "COMPLETED"
    assert response["message"] == "Research completed."
    assert isinstance(response["result"], dict)



def test_run_research_request_calls_research_pipeline(monkeypatch):
    calls = {}

    def fake_get_model():
        calls["model"] = True
        return FakeModel()

    def fake_budget_from_env():
        calls["budget"] = True
        return "fake-budget"

    def fake_run_research(question, model, budget):
        calls["question"] = question
        calls["model_value"] = model
        calls["budget_value"] = budget
        return FakeResult()

    monkeypatch.setattr(
        api,
        "get_model",
        fake_get_model,
    )

    monkeypatch.setattr(
        api.Budget,
        "from_env",
        staticmethod(fake_budget_from_env),
    )

    monkeypatch.setattr(
        api,
        "run_research",
        fake_run_research,
    )

    response = api.run_research_request(
        "  What are the applications of AI?  "
    )

    assert calls["model"] is True
    assert calls["budget"] is True
    assert calls["question"] == "What are the applications of AI?"
    assert isinstance(calls["model_value"], FakeModel)
    assert calls["budget_value"] == "fake-budget"

    assert response["question"] == "What are the applications of AI?"
    assert response["model"] == "fake:test-model"
    assert response["status"] == "COMPLETED"
    assert response["message"] == "Research completed."


def test_run_research_request_rejects_invalid_question(monkeypatch):
    def fail_if_called():
        raise AssertionError("Model should not be created")

    monkeypatch.setattr(
        api,
        "get_model",
        fail_if_called,
    )

    with pytest.raises(
        ValueError,
        match="Please enter a research question",
    ):
        api.run_research_request("   ")


def test_run_research_request_rejects_oversized_question(monkeypatch):
    def fail_if_called():
        raise AssertionError("Model should not be created")

    monkeypatch.setattr(
        api,
        "get_model",
        fail_if_called,
    )

    question = "x" * (api.MAX_QUESTION_LENGTH + 1)

    with pytest.raises(
        ValueError,
        match="Maximum length is 2000 characters",
    ):
        api.run_research_request(question)