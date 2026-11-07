from functools import lru_cache
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from backend.app.agent.prompts import SYSTEM_PROMPT
from backend.app.core.config import get_settings
from backend.app.tools.jokes import create_joke_tools

from opentelemetry import trace
tracer = trace.get_tracer("joke-agent-lab.agent")

def _message_to_text(message: object) -> str:
    """Extract assistant-readable text from a LangChain message."""
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                text_value = block.get("text")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
        return "\n".join(part for part in text_parts if part).strip()

    return str(content)


def _create_runtime() -> tuple[object, InMemorySaver]:
    """Build local Ollama chat models, tools, graph, and in-memory checkpointer."""
    settings = get_settings()
    base_url = settings.ollama_base_url.rstrip("/")
    api_key = settings.ollama_api_key.get_secret_value()

    # Ollama exposes an OpenAI-compatible Chat Completions endpoint at {base_url}/chat/completions.
    # The local server ignores the placeholder API key; no OpenAI-hosted API is called.
    agent_llm = ChatOpenAI(
        model=settings.ollama_model,
        api_key=api_key,
        base_url=base_url,
        temperature=settings.agent_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
    )

    # Use the same local model with a higher temperature for more creative joke generation.
    joke_llm = ChatOpenAI(
        model=settings.ollama_model,
        api_key=api_key,
        base_url=base_url,
        temperature=settings.joke_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
    )

    tools = create_joke_tools(joke_llm)
    agent_llm_with_tools = agent_llm.bind_tools(tools)

    def agent_node(state: MessagesState) -> dict[str, list[object]]:
        """Ask the local model for the next assistant message or tool call."""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"],
        ]
        response = agent_llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    checkpointer = InMemorySaver()
    compiled_graph = builder.compile(checkpointer=checkpointer)
    return compiled_graph, checkpointer


@lru_cache(maxsize=1)
def _get_runtime() -> tuple[object, InMemorySaver]:
    """Initialize the agent runtime only once per backend process."""
    return _create_runtime()


def chat(session_id: UUID, message: str) -> str:
    """Run one user turn while preserving state for the given session UUID."""
    graph, _ = _get_runtime()
    settings = get_settings()

    config = {
        "configurable": {"thread_id": str(session_id)},
        # This is a top-level graph config option, not part of `configurable`.
        "recursion_limit": settings.graph_recursion_limit,
    }

    with tracer.start_as_current_span("joke_agent.chat") as span:
        span.set_attribute("agent.framework", "langgraph")
        span.set_attribute("agent.operation", "chat")

        result = graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )

    messages = result.get("messages", [])
    if not messages:
        raise RuntimeError("The agent graph returned no messages.")

    answer = _message_to_text(messages[-1])
    if not answer:
        raise RuntimeError("The agent graph returned an empty assistant response.")

    return answer


def reset_chat(session_id: UUID) -> None:
    """Delete all checkpoints and pending writes for one conversation."""
    # If no chat request has initialized the graph, there is no checkpoint to delete.
    if _get_runtime.cache_info().currsize == 0:
        return

    _, checkpointer = _get_runtime()
    checkpointer.delete_thread(str(session_id))