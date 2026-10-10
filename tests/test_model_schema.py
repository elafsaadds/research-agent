from research_agent.model import OllamaModel, get_model


def test_schema_flag_enables_constrained_output(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("STRUCTURED_OUTPUT", "1")
    m = get_model()
    assert isinstance(m, OllamaModel)
    assert m.response_schema is not None
    assert m.name.endswith("+schema")


def test_schema_off_by_default(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.delenv("STRUCTURED_OUTPUT", raising=False)
    assert get_model().response_schema is None
