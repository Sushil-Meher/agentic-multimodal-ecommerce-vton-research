from sqlalchemy import text
from sqlalchemy.engine import Connection

from .products import _BASE_COLUMNS


def get_product_details(conn: Connection, sku: str, *, include_price: bool = True) -> dict | None:
    """Full lookup for one active product by SKU. Used by the agent tool,
    which needs price (a public endpoint would use catalog.public_products
    instead, same as the rest of the API)."""
    columns = _BASE_COLUMNS + (", pr.price, pr.currency" if include_price else "")
    query = f"""
        SELECT {columns}
        FROM catalog.products p
        JOIN commerce.product_prices pr ON pr.product_id = p.id AND pr.is_current
        WHERE p.sku = :sku AND p.status = 'ACTIVE'
    """
    row = conn.execute(text(query), {"sku": sku}).mappings().first()
    return dict(row) if row else None
