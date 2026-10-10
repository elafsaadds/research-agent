from research_agent.baseline import load_questions
from research_agent.model import MockModel, get_model


def test_mock_model_is_deterministic():
    m = MockModel()
    msgs = [{"role": "user", "content": "hello"}]
    assert m.generate(msgs).text == m.generate(msgs).text


def test_get_model_defaults_to_mock(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "mock")
    assert get_model().name == "mock"


def test_frozen_questions_are_valid():
    qs = load_questions()
    assert len(qs) == 8
    assert len({q["id"] for q in qs}) == 8
    assert all(q["expected_points"] for q in qs)
