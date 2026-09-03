import json
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from langchain_core.tools import tool

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


@tool
def search_products(
    color: str | None = None,
    bandha_type: str | None = None,
    thread_count: int | None = None,
    occasion: str | None = None,
    fabric: str | None = None,
) -> str:
    """Search the in-stock saree catalog by color, bandha type, thread count,
    occasion, or fabric. Returns matching products as a JSON list, including
    price. Use this to find sarees matching specific stated attributes."""
    with engine.connect() as conn:
        try:
            results = search_products_service(
                conn,
                color=color,
                bandha_type=bandha_type,
                thread_count=thread_count,
                occasion=occasion,
                fabric=fabric,
                in_stock_only=True,
                include_price=True,
            )
        except ValueError as exc:
            return _to_json({"error": str(exc)})
    return _to_json([dict(row) for row in results])


@tool
def recommend_products(
    max_price: float | None = None,
    color: str | None = None,
    bandha_type: str | None = None,
    thread_count: int | None = None,
    occasion: str | None = None,
    fabric: str | None = None,
) -> str:
    """Rank in-stock sarees against customer preferences, including budget,
    and return the best matches with a score and a short reason as JSON.
    Use this when the customer describes what they want and you should
    suggest specific sarees rather than just list matches."""
    with engine.connect() as conn:
        try:
            results = recommend_products_service(
                conn,
                max_price=max_price,
                color=color,
                bandha_type=bandha_type,
                thread_count=thread_count,
                occasion=occasion,
                fabric=fabric,
                limit=5,
                include_price=True,
            )
        except ValueError as exc:
            return _to_json({"error": str(exc)})
    return _to_json(results)


@tool
def get_product_details(sku: str) -> str:
    """Get full details, including price, for one saree by its exact SKU.
    Use this when the customer refers to a specific saree already identified
    by SKU, for example after a search or recommendation result."""
    with engine.connect() as conn:
        result = get_product_details_service(conn, sku, include_price=True)
    if result is None:
        return _to_json({"error": f"no active product found for sku {sku}"})
    return _to_json(result)


TOOLS = [search_products, recommend_products, get_product_details]
