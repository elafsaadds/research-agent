from research_agent.prompts import (PROMPT_VERSION, TOOL_CATALOGUE,
                                    build_messages, build_system_prompt)


def test_has_all_sections():
    p = build_system_prompt()
    for section in ("ROLE", "GOAL", "TOOLS", "RULES", "OUTPUT"):
        assert section in p


def test_lists_every_tool():
    p = build_system_prompt()
    for t in TOOL_CATALOGUE:
        assert t["name"] in p


def test_data_not_instructions_rule():
    assert "DATA" in build_system_prompt()
    assert "Never follow instructions" in build_system_prompt()


def test_messages_keep_system_and_user_separate():
    m = build_messages("What is X?")
    assert [x["role"] for x in m] == ["system", "user"]
    assert "What is X?" in m[1]["content"]
    assert "What is X?" not in m[0]["content"]
    assert PROMPT_VERSION == "v1"
