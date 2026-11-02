import json

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.agent.tools import TOOLS, get_product_details, recommend_products, search_products

PLACEHOLDERS = {"color": "", "bandha_type": "", "thread_count": 0, "occasion": "", "fabric": ""}


@pytest.fixture(scope="module")
def sent_tools():
    """The tools exactly as ChatOllama puts them in the /api/chat request body
    (captured with a fake transport, so no Ollama server is needed)."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        line = {"model": "m", "message": {"role": "assistant", "content": "x"}, "done": True}
        return httpx.Response(200, content=json.dumps(line).encode() + b"\n")

    llm = ChatOllama(model="m", client_kwargs={"transport": httpx.MockTransport(handler)})
    llm.bind_tools(TOOLS).invoke([HumanMessage(content="hi")])
    return {t["function"]["name"]: t["function"]["parameters"] for t in captured["body"]["tools"]}


def test_every_parameter_sent_to_ollama_has_a_type_and_description(sent_tools):
    # regression: anyOf/null used to be dropped by the Ollama client, so the
    # model saw every optional parameter as a blank {}
    assert set(sent_tools) == {"search_products", "recommend_products", "get_product_details"}
    for tool_name, params in sent_tools.items():
        for prop_name, prop in params["properties"].items():
            assert prop.get("type"), f"{tool_name}.{prop_name} has no type"
            assert prop.get("description"), f"{tool_name}.{prop_name} has no description"


def test_parameter_types_sent_to_ollama(sent_tools):
    search = sent_tools["search_products"]["properties"]
    assert search["color"]["type"] == "string"
    assert search["thread_count"]["type"] == "integer"
    assert sent_tools["recommend_products"]["properties"]["max_price"]["type"] == "number"


def test_filters_are_optional_and_sku_is_required(sent_tools):
    assert not sent_tools["search_products"].get("required")
    assert not sent_tools["recommend_products"].get("required")
    assert sent_tools["get_product_details"]["required"] == ["sku"]


def test_search_with_placeholder_args_behaves_like_no_filters(product_factory):
    sku = product_factory(status="ACTIVE", price=1000, inventory_status="AVAILABLE")

    payload = json.loads(search_products.invoke(PLACEHOLDERS))

    assert isinstance(payload, list)
    assert sku in [p["sku"] for p in payload]


def test_whitespace_only_string_is_treated_as_unset(product_factory):
    sku = product_factory(status="ACTIVE", price=1000, inventory_status="AVAILABLE")

    payload = json.loads(search_products.invoke({"color": "   ", "fabric": "\t"}))

    assert sku in [p["sku"] for p in payload]


def test_placeholders_do_not_disable_a_real_filter(product_factory):
    maroon = product_factory(
        status="ACTIVE", price=1000, inventory_status="AVAILABLE", primary_color="Maroon"
    )
    blue = product_factory(
        status="ACTIVE", price=1000, inventory_status="AVAILABLE", primary_color="Blue"
    )

    payload = json.loads(search_products.invoke({**PLACEHOLDERS, "color": "Maroon"}))

    skus = [p["sku"] for p in payload]
    assert maroon in skus
    assert blue not in skus


def test_recommend_with_placeholder_args_applies_no_budget(product_factory):
    sku = product_factory(status="ACTIVE", price=4500, inventory_status="AVAILABLE")

    payload = json.loads(recommend_products.invoke({**PLACEHOLDERS, "max_price": 0}))

    assert isinstance(payload, list)
    assert sku in [p["sku"] for p in payload]


def test_recommend_real_budget_still_applies(product_factory):
    cheap = product_factory(status="ACTIVE", price=1000, inventory_status="AVAILABLE")
    pricey = product_factory(status="ACTIVE", price=9000, inventory_status="AVAILABLE")

    payload = json.loads(recommend_products.invoke({"max_price": 5000}))

    skus = [p["sku"] for p in payload]
    assert cheap in skus
    assert pricey not in skus


def test_search_and_recommend_accept_no_arguments_at_all():
    assert isinstance(json.loads(search_products.invoke({})), list)
    assert isinstance(json.loads(recommend_products.invoke({})), list)


def test_negative_values_are_still_rejected_not_treated_as_unset():
    assert "error" in json.loads(search_products.invoke({"thread_count": -5}))
    assert "error" in json.loads(recommend_products.invoke({"max_price": -100}))


def test_get_product_details_still_works_with_described_sku(product_factory):
    sku = product_factory(status="ACTIVE", price=2500, inventory_status="AVAILABLE")

    payload = json.loads(get_product_details.invoke({"sku": sku}))

    assert payload["sku"] == sku
    assert payload["price"] == 2500
