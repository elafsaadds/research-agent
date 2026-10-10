from research_agent.schemas import decision_json_schema


def test_schema_variants_make_invalid_mixes_unrepresentable():
    tool, other = decision_json_schema()["anyOf"]
    assert "message" not in tool["properties"]
    assert "tool_name" not in other["properties"]
    assert "arguments" not in other["properties"]
    assert set(other["properties"]["action"]["enum"]) == {
        "answer", "ask_user", "insufficient_evidence"}
    assert tool["additionalProperties"] is False
