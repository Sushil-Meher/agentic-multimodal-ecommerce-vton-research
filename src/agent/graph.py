import json
import os
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
    "that came from a tool result, never invent them. Keep answers short. "
    "Always run a search before saying you cannot find, filter or recognise something; "
    "never decide a word is unknown without searching for it first. "
    "Search only for the colours and designs the customer named; never add other colours or "
    "words to a search to widen it. If nothing matches, say so and offer to try something else. "
    "Refer to each saree by its name (for example BunkarHouse00017), never by SKU. "
    "If a customer asks for something the filters cannot express, say you cannot "
    "filter for that; never say none exist unless a search with the right filters "
    "found none. When a result has colour_match 'weaker', tell the customer honestly "
    "using its colour_note."
)


def load_glossary() -> str:
    """Store-specific terms (owner's Sambalpuri words) so the model searches them the right way.
    The file stays outside the repo; without it the agent simply has no glossary."""
    path = os.getenv("AGENT_GLOSSARY_FILE", r"D:\saree_dataset\glossary.json")
    try:
        with open(path, encoding="utf-8") as f:
            terms = json.load(f)["terms"]
    except (OSError, ValueError, KeyError):
        return ""
    lines = []
    for t in terms:
        also = f" (also written {', '.join(t['aliases'])})" if t.get("aliases") else ""
        lines.append(f'- "{t["term"]}"{also}: {t["meaning"]}')
    return "\n\nStore terms (use the term exactly as written in the first quotes when searching):\n" + "\n".join(lines)


class AgentState(TypedDict):
    # every user/agent/tool turn is one message; add_messages appends new
    # ones instead of overwriting, which is the standard LangGraph pattern
    messages: Annotated[list, add_messages]


def build_graph(llm=None):
    """Builds the sales agent graph. Pass llm= a fake model in tests so the
    graph runs without a real API key or network call."""
    llm = (llm or get_llm()).bind_tools(TOOLS)
    system_prompt = SYSTEM_PROMPT + load_glossary()

    def call_model(state: AgentState):
        response = llm.invoke([SystemMessage(content=system_prompt), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "call_model")
    return graph.compile()
