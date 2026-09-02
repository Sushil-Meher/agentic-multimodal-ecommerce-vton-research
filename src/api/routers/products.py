from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.engine import Connection

from ..db import get_connection
from ..schemas import Product

router = APIRouter(prefix="/products", tags=["products"])

# always reads from the public view, never the base products table,
# so private columns can't leak here even by accident
_BASE_QUERY = "SELECT * FROM catalog.public_products"


@router.get("", response_model=list[Product])
def list_products(conn: Connection = Depends(get_connection)):
    rows = conn.execute(text(f"{_BASE_QUERY} ORDER BY created_at DESC")).mappings().all()
    return rows


@router.get("/search", response_model=list[Product])
def search_products(
    color: str | None = None,
    bandha_type: str | None = None,
    max_thread_count: int | None = Query(default=None, gt=0),
    conn: Connection = Depends(get_connection),
):
    clauses = []
    params: dict[str, str | int] = {}

    if color:
        clauses.append("primary_color ILIKE :color")
        params["color"] = color
    if bandha_type:
        clauses.append("bandha_type ILIKE :bandha_type")
        params["bandha_type"] = bandha_type
    if max_thread_count is not None:
        clauses.append("thread_count <= :max_thread_count")
        params["max_thread_count"] = max_thread_count

    query = _BASE_QUERY
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC"

    rows = conn.execute(text(query), params).mappings().all()
    return rows


@router.get("/{sku}", response_model=Product)
def get_product(sku: str, conn: Connection = Depends(get_connection)):
    row = conn.execute(text(f"{_BASE_QUERY} WHERE sku = :sku"), {"sku": sku}).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="product not found")
    return row
