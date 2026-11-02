import json
import os
from datetime import datetime
from decimal import Decimal
from typing import Annotated
from uuid import UUID

from langchain_core.tools import tool
from pydantic import Field

from ..api.db import engine
from ..search.details import get_product_details as get_product_details_service
from ..search.products import search_products as search_products_service
from ..search.recommend import recommend_products as recommend_products_service


def _json_default(value):
    if isinstance(value, (UUID, datetime)):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"not JSON serializable: {value!r}")


def _to_json(data) -> str:
    return json.dumps(data, default=_json_default)


def _include_eval_fixtures() -> bool:
    # the evaluation harness sets this so its own EVAL-* fixture catalog is
    # reachable through the real tools; real customers never set it
    return os.getenv("AGENT_INCLUDE_EVAL_FIXTURES") == "1"


# Optional filters use plain types with "" / 0 as "not given" instead of
# `X | None`: the Ollama client drops anyOf/null from tool schemas, so the
# model used to see every optional parameter as a blank `{}`.
Color = Annotated[
    str, Field(description="Primary color, e.g. Maroon. Omit if the customer didn't mention one.")
]
BandhaType = Annotated[
    str,
    Field(
        description="Full bandha style name, e.g. 'Traditional Bandha' or 'Pasapalli Bandha'. "
        "Omit if the customer didn't mention one."
    ),
]
ThreadCount = Annotated[
    int,
    Field(description="Thread count as a positive integer, e.g. 100. Omit if not mentioned."),
]
Occasion = Annotated[
    str, Field(description="Occasion, e.g. Wedding or Festival. Omit if not mentioned.")
]
Fabric = Annotated[str, Field(description="Fabric, e.g. Cotton or Silk. Omit if not mentioned.")]
MaxPrice = Annotated[
    float,
    Field(description="Budget ceiling in rupees. Omit if the customer didn't state a budget."),
]


def _unset(value):
    # a model may still send "" or 0 for an argument it should have omitted
    if isinstance(value, str):
        value = value.strip()
    return None if value in ("", 0, None) else value


@tool
def search_products(
    color: Color = "",
    bandha_type: BandhaType = "",
    thread_count: ThreadCount = 0,
    occasion: Occasion = "",
    fabric: Fabric = "",
) -> str:
    """Search the in-stock saree catalog by color, bandha type, thread count,
    occasion, or fabric. Returns matching products as a JSON list, including
    price. Pass only the filters the customer stated; call with no arguments
    to list everything in stock."""
    with engine.connect() as conn:
        try:
            results = search_products_service(
                conn,
                color=_unset(color),
                bandha_type=_unset(bandha_type),
                thread_count=_unset(thread_count),
                occasion=_unset(occasion),
                fabric=_unset(fabric),
                in_stock_only=True,
                include_price=True,
                include_eval_fixtures=_include_eval_fixtures(),
            )
        except ValueError as exc:
            return _to_json({"error": str(exc)})
    return _to_json([dict(row) for row in results])


@tool
def recommend_products(
    max_price: MaxPrice = 0,
    color: Color = "",
    bandha_type: BandhaType = "",
    thread_count: ThreadCount = 0,
    occasion: Occasion = "",
    fabric: Fabric = "",
) -> str:
    """Rank in-stock sarees against customer preferences, including budget,
    and return the best matches with a score and a short reason as JSON.
    Use this when the customer describes what they want and you should
    suggest specific sarees rather than just list matches. Pass only the
    preferences the customer stated."""
    with engine.connect() as conn:
        try:
            results = recommend_products_service(
                conn,
                max_price=_unset(max_price),
                color=_unset(color),
                bandha_type=_unset(bandha_type),
                thread_count=_unset(thread_count),
                occasion=_unset(occasion),
                fabric=_unset(fabric),
                limit=5,
                include_price=True,
                include_eval_fixtures=_include_eval_fixtures(),
            )
        except ValueError as exc:
            return _to_json({"error": str(exc)})
    return _to_json(results)


@tool
def get_product_details(
    sku: Annotated[str, Field(description="Exact product SKU, e.g. SAM-2026-0001.")],
) -> str:
    """Get full details, including price, for one saree by its exact SKU.
    Use this when the customer refers to a specific saree already identified
    by SKU, for example after a search or recommendation result."""
    with engine.connect() as conn:
        result = get_product_details_service(
            conn, sku, include_price=True, include_eval_fixtures=_include_eval_fixtures()
        )
    if result is None:
        return _to_json({"error": f"no active product found for sku {sku}"})
    return _to_json(result)


TOOLS = [search_products, recommend_products, get_product_details]
