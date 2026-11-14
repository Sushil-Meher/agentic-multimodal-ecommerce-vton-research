from sqlalchemy import text
from sqlalchemy.engine import Connection

from .attributes import colour_match, design_clauses, normalise_blouse, normalise_region, parse_terms

# evaluation-harness fixtures use this SKU prefix (see experiments/); never
# real catalog products, so callers must opt in to see them
EVAL_FIXTURE_SKU_PREFIX = "EVAL-"

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
    include_eval_fixtures: bool = False,
    colors: str | None = None,
    blouse: str | None = None,
    design: str | None = None,
    region: str | None = None,
    include_draft: bool = False,
) -> list[dict]:
    """Reusable product search. Meant to be called from the API layer today
    and directly from an agent tool later, so filters stay simple keyword
    arguments rather than a request-shaped object.

    colors: free text such as "red" or "red and black" (see attributes.colour_match);
    blouse: plain / blouse work / no blouse; design: one or more words matched
    against pattern labels and motifs; region: body / border / pallu, narrows design.
    include_draft is for development only (unpublished sarees)."""

    if max_price is not None and max_price < 0:
        raise ValueError("max_price must be >= 0")
    if min_price is not None and min_price < 0:
        raise ValueError("min_price must be >= 0")
    if max_price is not None and min_price is not None and min_price > max_price:
        raise ValueError("min_price cannot be greater than max_price")
    if thread_count is not None and thread_count <= 0:
        raise ValueError("thread_count must be > 0")

    wanted_colors = parse_terms(colors)
    designs = parse_terms(design)
    blouse = normalise_blouse(blouse) if blouse else None
    region = normalise_region(region) if region else None

    clauses = ["p.status IN ('ACTIVE', 'DRAFT')" if include_draft else "p.status = 'ACTIVE'"]
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
    if blouse:
        clauses.append(
            "EXISTS (SELECT 1 FROM catalog.product_attributes b WHERE b.product_id = p.id "
            "AND b.attribute = 'blouse' AND b.value = :blouse)"
        )
        params["blouse"] = blouse
    clauses += design_clauses(designs, region, params)
    if in_stock_only:
        clauses.append(_AVAILABLE_EXISTS)
    if not include_eval_fixtures:
        clauses.append("p.sku NOT LIKE :eval_prefix")
        params["eval_prefix"] = f"{EVAL_FIXTURE_SKU_PREFIX}%"

    columns = _BASE_COLUMNS + (", pr.price, pr.currency" if include_price else "")
    if wanted_colors:
        columns += """, COALESCE((SELECT array_agg(c.value) FROM catalog.product_attributes c
                         WHERE c.product_id = p.id AND c.attribute = 'colour_other'), '{}') AS body_other_colours"""

    # current price is joined even when not selected, so a product with no
    # price yet (not really for sale) doesn't show up in search results
    query = f"""
        SELECT {columns}
        FROM catalog.products p
        JOIN commerce.product_prices pr ON pr.product_id = p.id AND pr.is_current
        WHERE {' AND '.join(clauses)}
        ORDER BY p.created_at DESC
    """

    rows = conn.execute(text(query), params).mappings().all()
    if not wanted_colors:
        return rows

    matched = []
    for row in rows:
        row = dict(row)
        found = colour_match(row["primary_color"], row["secondary_color"], row.pop("body_other_colours"), wanted_colors)
        if found:
            row["colour_match"], row["colour_note"] = found
            matched.append(row)
    matched.sort(key=lambda r: r["colour_match"] != "best")  # stable: best matches first
    return matched
