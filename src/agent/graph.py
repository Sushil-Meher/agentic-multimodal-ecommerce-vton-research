from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from .llm import get_llm
from .tools import TOOLS

SYSTEM_PROMPT = (
    "You are a sales assistant for a Sambalpuri handloom saree store. "
    "Use the available tools to search, recommend, or look up sarees before "
    "answering a product question. Only state product details and prices "
    "that came from a tool result, never invent them. Keep answers short."
)


class AgentState(TypedDict):
    # every user/agent/tool turn is one message; add_messages appends new
    # ones instead of overwriting, which is the standard LangGraph pattern
    messages: Annotated[list, add_messages]


def build_graph(llm=None):
    """Builds the sales agent graph. Pass llm= a fake model in tests so the
    graph runs without a real API key or network call."""
    llm = (llm or get_llm()).bind_tools(TOOLS)

    def call_model(state: AgentState):
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "call_model")
    return graph.compile()
