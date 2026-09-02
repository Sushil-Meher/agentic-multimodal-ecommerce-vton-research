from sqlalchemy import text
from sqlalchemy.engine import Connection

# columns every caller gets; price/currency are added only when asked for
_BASE_COLUMNS = """
    p.id, p.sku, p.name, p.description, p.category, p.fabric,
    p.primary_color, p.secondary_color, p.bandha_type, p.motif,
    p.thread_count, p.border_type, p.saree_length_cm, p.blouse_included,
    p.weaving_region, p.occasion, p.weight_grams, p.public_slug, p.status,
    p.created_at, p.updated_at
"""

_AVAILABLE_EXISTS = """
    EXISTS (
        SELECT 1 FROM commerce.inventory_items ii
        WHERE ii.product_id = p.id AND ii.status = 'AVAILABLE'
    )
"""


def search_products(
    conn: Connection,
    *,
    max_price: float | None = None,
    min_price: float | None = None,
    color: str | None = None,
    bandha_type: str | None = None,
    thread_count: int | None = None,
    occasion: str | None = None,
    fabric: str | None = None,
    in_stock_only: bool = True,
    include_price: bool = True,
) -> list[dict]:
    """Reusable product search. Meant to be called from the API layer today
    and directly from an agent tool later, so filters stay simple keyword
    arguments rather than a request-shaped object."""

    if max_price is not None and max_price < 0:
        raise ValueError("max_price must be >= 0")
    if min_price is not None and min_price < 0:
        raise ValueError("min_price must be >= 0")
    if max_price is not None and min_price is not None and min_price > max_price:
        raise ValueError("min_price cannot be greater than max_price")
    if thread_count is not None and thread_count <= 0:
        raise ValueError("thread_count must be > 0")

    clauses = ["p.status = 'ACTIVE'"]
    params: dict[str, object] = {}

    if max_price is not None:
        clauses.append("pr.price <= :max_price")
        params["max_price"] = max_price
    if min_price is not None:
        clauses.append("pr.price >= :min_price")
        params["min_price"] = min_price
    if color:
        clauses.append("p.primary_color ILIKE :color")
        params["color"] = color
    if bandha_type:
        clauses.append("p.bandha_type ILIKE :bandha_type")
        params["bandha_type"] = bandha_type
    if thread_count is not None:
        clauses.append("p.thread_count = :thread_count")
        params["thread_count"] = thread_count
    if occasion:
        clauses.append("p.occasion ILIKE :occasion")
        params["occasion"] = occasion
    if fabric:
        clauses.append("p.fabric ILIKE :fabric")
        params["fabric"] = fabric
    if in_stock_only:
        clauses.append(_AVAILABLE_EXISTS)

    columns = _BASE_COLUMNS + (", pr.price, pr.currency" if include_price else "")

    # current price is joined even when not selected, so a product with no
    # price yet (not really for sale) doesn't show up in search results
    query = f"""
        SELECT {columns}
        FROM catalog.products p
        JOIN commerce.product_prices pr ON pr.product_id = p.id AND pr.is_current
        WHERE {' AND '.join(clauses)}
        ORDER BY p.created_at DESC
    """

    return conn.execute(text(query), params).mappings().all()
