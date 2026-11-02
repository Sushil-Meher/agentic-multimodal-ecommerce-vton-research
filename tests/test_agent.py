import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import text

from src.agent.graph import build_graph
from src.agent.tools import TOOLS, get_product_details, recommend_products, search_products
from src.api.db import engine


class FakeLLM:
    """Stands in for a real chat model: returns canned responses in order
    instead of calling an LLM API, so the graph can be tested for free."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.invocations = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        return self._responses.pop(0)


def _tool_call(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def test_build_graph_and_basic_state_flow():
    llm = FakeLLM([AIMessage(content="Hello! How can I help you find a saree today?")])
    graph = build_graph(llm=llm)

    result = graph.invoke({"messages": [HumanMessage(content="hi")]})

    assert result["messages"][-1].content == "Hello! How can I help you find a saree today?"
    assert len(llm.invocations) == 1


def test_tool_invocation_appends_tool_message_and_loops_back(product_factory):
    sku = product_factory(
        status="ACTIVE", price=1000, inventory_status="AVAILABLE", primary_color="Maroon"
    )

    llm = FakeLLM(
        [
            _tool_call("search_products", {"color": "Maroon"}),
            AIMessage(content="Found a maroon saree for you."),
        ]
    )
    graph = build_graph(llm=llm)

    result = graph.invoke({"messages": [HumanMessage(content="show me maroon sarees")]})
    messages = result["messages"]

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert sku in tool_messages[0].content
    assert messages[-1].content == "Found a maroon saree for you."
    assert len(llm.invocations) == 2


def test_product_search_request(product_factory):
    sku = product_factory(
        status="ACTIVE", price=1500, inventory_status="AVAILABLE", occasion="Wedding"
    )

    llm = FakeLLM(
        [
            _tool_call("search_products", {"occasion": "Wedding"}),
            AIMessage(content="Here are wedding sarees."),
        ]
    )
    graph = build_graph(llm=llm)

    result = graph.invoke({"messages": [HumanMessage(content="wedding sarees please")]})
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    payload = json.loads(tool_message.content)

    assert any(p["sku"] == sku for p in payload)


def test_recommendation_request(product_factory):
    sku = product_factory(
        status="ACTIVE",
        price=2000,
        inventory_status="AVAILABLE",
        primary_color="Maroon",
        occasion="Wedding",
    )

    llm = FakeLLM(
        [
            _tool_call("recommend_products", {"color": "Maroon", "occasion": "Wedding"}),
            AIMessage(content="This one matches best."),
        ]
    )
    graph = build_graph(llm=llm)

    result = graph.invoke({"messages": [HumanMessage(content="what matches maroon wedding?")]})
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    payload = json.loads(tool_message.content)
    match = next(p for p in payload if p["sku"] == sku)

    assert match["score"] > 0
    assert "reason" in match


def test_product_details_request(product_factory):
    sku = product_factory(status="ACTIVE", price=2500, inventory_status="AVAILABLE")

    llm = FakeLLM(
        [
            _tool_call("get_product_details", {"sku": sku}),
            AIMessage(content="Here are the details."),
        ]
    )
    graph = build_graph(llm=llm)

    result = graph.invoke({"messages": [HumanMessage(content=f"tell me about {sku}")]})
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    payload = json.loads(tool_message.content)

    assert payload["sku"] == sku
    assert payload["price"] == 2500


def test_ollama_style_recommendation_flow(product_factory):
    # ChatOllama returns tool calls in the same AIMessage.tool_calls shape
    # as every other LangChain chat model, so FakeLLM here stands in for
    # a real local Qwen3-8B call without needing Ollama running for tests.
    sku = product_factory(
        status="ACTIVE",
        price=4500,
        inventory_status="AVAILABLE",
        primary_color="Maroon",
        occasion="Wedding",
    )

    llm = FakeLLM(
        [
            _tool_call("recommend_products", {"color": "Maroon", "occasion": "Wedding", "max_price": 5000}),
            AIMessage(content="The maroon wedding saree fits your budget."),
        ]
    )
    graph = build_graph(llm=llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="maroon saree for a wedding, budget 5000")]}
    )
    messages = result["messages"]

    tool_message = next(m for m in messages if isinstance(m, ToolMessage))
    payload = json.loads(tool_message.content)
    assert any(p["sku"] == sku for p in payload)
    assert messages[-1].content == "The maroon wedding saree fits your budget."


def test_invalid_tool_input_returns_error_payload_not_a_crash():
    payload = json.loads(search_products.invoke({"thread_count": -5}))
    assert "error" in payload


def test_no_result_response():
    payload = json.loads(recommend_products.invoke({"max_price": 1}))
    assert payload == []


def test_get_product_details_unknown_sku_returns_error():
    payload = json.loads(get_product_details.invoke({"sku": "DOES-NOT-EXIST"}))
    assert "error" in payload


def test_agent_tool_excludes_eval_fixture_by_default(product_factory, monkeypatch):
    monkeypatch.delenv("AGENT_INCLUDE_EVAL_FIXTURES", raising=False)
    sku = product_factory(
        status="ACTIVE", price=1000, inventory_status="AVAILABLE",
        sku="EVAL-TEST-0006", primary_color="Maroon",
    )

    payload = json.loads(search_products.invoke({"color": "Maroon"}))

    assert sku not in [p["sku"] for p in payload]


def test_agent_tool_includes_eval_fixture_when_enabled(product_factory, monkeypatch):
    monkeypatch.setenv("AGENT_INCLUDE_EVAL_FIXTURES", "1")
    sku = product_factory(
        status="ACTIVE", price=1000, inventory_status="AVAILABLE",
        sku="EVAL-TEST-0007", primary_color="Maroon",
    )

    payload = json.loads(search_products.invoke({"color": "Maroon"}))

    assert sku in [p["sku"] for p in payload]


def test_tool_list_has_no_raw_sql_tool():
    names = {t.name for t in TOOLS}
    assert names == {"search_products", "recommend_products", "get_product_details"}


def test_search_tool_is_immune_to_sql_injection_in_arguments():
    with engine.connect() as conn:
        before = conn.execute(text("SELECT count(*) FROM catalog.products")).scalar()

    payload = json.loads(search_products.invoke({"color": "Maroon'; DROP TABLE catalog.products; --"}))

    with engine.connect() as conn:
        after = conn.execute(text("SELECT count(*) FROM catalog.products")).scalar()

    assert payload == []
    assert after == before
