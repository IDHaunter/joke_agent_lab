import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from backend.app.tools.jokes import create_joke_tools


class FakeLLM:
    """Small stand-in for a chat model; no network call is made by these tests."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> AIMessage:
        self.prompts.append(prompt)
        return AIMessage(content="A generated test joke.")


def test_three_tools_are_created() -> None:
    llm = FakeLLM()
    tools = create_joke_tools(llm)  # type: ignore[arg-type]

    assert [tool.name for tool in tools] == [
        "tell_ussr_joke",
        "tell_stirlitz_joke",
        "tell_capitalism_joke",
    ]


def test_tool_passes_count_language_and_keywords_to_generation_prompt() -> None:
    llm = FakeLLM()
    tools = create_joke_tools(llm)  # type: ignore[arg-type]
    tool_by_name = {tool.name: tool for tool in tools}

    result = tool_by_name["tell_stirlitz_joke"].invoke(
        {
            "count": 2,
            "language": "English",
            "keywords": ["train station", "briefcase", "secret message"],
        }
    )

    assert result == "A generated test joke."
    assert len(llm.prompts) == 1
    assert "exactly 2" in llm.prompts[0]
    assert "Language: English" in llm.prompts[0]
    assert "Stirlitz" in llm.prompts[0]
    assert "- train station" in llm.prompts[0]
    assert "- briefcase" in llm.prompts[0]
    assert "- secret message" in llm.prompts[0]


def test_tool_accepts_empty_keywords() -> None:
    llm = FakeLLM()
    tools = create_joke_tools(llm)  # type: ignore[arg-type]
    tool_by_name = {tool.name: tool for tool in tools}

    result = tool_by_name["tell_ussr_joke"].invoke(
        {"count": 1, "language": "Russian", "keywords": []}
    )

    assert result == "A generated test joke."
    assert "No specific keywords were provided" in llm.prompts[0]


def test_tool_rejects_count_above_five() -> None:
    llm = FakeLLM()
    tools = create_joke_tools(llm)  # type: ignore[arg-type]
    tool_by_name = {tool.name: tool for tool in tools}

    with pytest.raises(ValidationError):
        tool_by_name["tell_ussr_joke"].invoke(
            {"count": 6, "language": "Russian", "keywords": []}
        )

    assert llm.prompts == []


def test_tool_rejects_count_below_one() -> None:
    llm = FakeLLM()
    tools = create_joke_tools(llm)  # type: ignore[arg-type]
    tool_by_name = {tool.name: tool for tool in tools}

    with pytest.raises(ValidationError):
        tool_by_name["tell_ussr_joke"].invoke(
            {"count": 0, "language": "Russian", "keywords": []}
        )

    assert llm.prompts == []


def test_tool_rejects_more_than_eight_keywords() -> None:
    llm = FakeLLM()
    tools = create_joke_tools(llm)  # type: ignore[arg-type]
    tool_by_name = {tool.name: tool for tool in tools}

    with pytest.raises(ValidationError):
        tool_by_name["tell_ussr_joke"].invoke(
            {
                "count": 1,
                "language": "Russian",
                "keywords": [f"keyword-{index}" for index in range(9)],
            }
        )

    assert llm.prompts == []


def test_tool_rejects_empty_keyword_string() -> None:
    llm = FakeLLM()
    tools = create_joke_tools(llm)  # type: ignore[arg-type]
    tool_by_name = {tool.name: tool for tool in tools}

    with pytest.raises(ValidationError):
        tool_by_name["tell_ussr_joke"].invoke(
            {"count": 1, "language": "Russian", "keywords": ["   "]}
        )

    assert llm.prompts == []


def test_tool_rejects_keyword_longer_than_eighty_characters() -> None:
    llm = FakeLLM()
    tools = create_joke_tools(llm)  # type: ignore[arg-type]
    tool_by_name = {tool.name: tool for tool in tools}

    with pytest.raises(ValidationError):
        tool_by_name["tell_ussr_joke"].invoke(
            {"count": 1, "language": "Russian", "keywords": ["x" * 81]}
        )

    assert llm.prompts == []


def test_tool_trims_and_deduplicates_keywords() -> None:
    llm = FakeLLM()
    tools = create_joke_tools(llm)  # type: ignore[arg-type]
    tool_by_name = {tool.name: tool for tool in tools}

    tool_by_name["tell_ussr_joke"].invoke(
        {
            "count": 1,
            "language": "Russian",
            "keywords": ["Brezhnev", " brezhnev ", "queues"],
        }
    )

    assert llm.prompts[0].count("- Brezhnev") == 1
    assert "- queues" in llm.prompts[0]