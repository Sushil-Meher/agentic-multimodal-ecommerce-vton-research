from sqlalchemy import text
from sqlalchemy.engine import Connection

from .products import _BASE_COLUMNS, EVAL_FIXTURE_SKU_PREFIX


def get_product_details(
    conn: Connection, sku: str, *, include_price: bool = True, include_eval_fixtures: bool = False
) -> dict | None:
    """Full lookup for one active product by SKU. Used by the agent tool,
    which needs price (a public endpoint would use catalog.public_products
    instead, same as the rest of the API)."""
    columns = _BASE_COLUMNS + (", pr.price, pr.currency" if include_price else "")
    eval_clause = "" if include_eval_fixtures else "AND p.sku NOT LIKE :eval_prefix"
    query = f"""
        SELECT {columns}
        FROM catalog.products p
        JOIN commerce.product_prices pr ON pr.product_id = p.id AND pr.is_current
        WHERE p.sku = :sku AND p.status = 'ACTIVE' {eval_clause}
    """
    params = {"sku": sku}
    if not include_eval_fixtures:
        params["eval_prefix"] = f"{EVAL_FIXTURE_SKU_PREFIX}%"
    row = conn.execute(text(query), params).mappings().first()
    return dict(row) if row else None
