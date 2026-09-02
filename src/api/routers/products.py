from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.engine import Connection

from ..db import get_connection
from ..schemas import Product, Recommendation
from ...search.products import search_products as search_products_service
from ...search.recommend import recommend_products as recommend_products_service

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
    thread_count: int | None = Query(default=None, gt=0),
    occasion: str | None = None,
    fabric: str | None = None,
    in_stock_only: bool = True,
    conn: Connection = Depends(get_connection),
):
    # no price filter here on purpose: even without ever returning a price,
    # letting callers bisect on max_price would leak it indirectly
    try:
        rows = search_products_service(
            conn,
            color=color,
            bandha_type=bandha_type,
            thread_count=thread_count,
            occasion=occasion,
            fabric=fabric,
            in_stock_only=in_stock_only,
            include_price=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return rows


@router.get("/recommendations", response_model=list[Recommendation])
def recommend_products(
    color: str | None = None,
    bandha_type: str | None = None,
    thread_count: int | None = Query(default=None, gt=0),
    occasion: str | None = None,
    fabric: str | None = None,
    limit: int = Query(default=5, gt=0, le=20),
    conn: Connection = Depends(get_connection),
):
    # same reasoning as /search: no public max_price, it would leak pricing
    # information indirectly even without returning a price
    try:
        results = recommend_products_service(
            conn,
            color=color,
            bandha_type=bandha_type,
            thread_count=thread_count,
            occasion=occasion,
            fabric=fabric,
            limit=limit,
            include_price=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return results


@router.get("/{sku}", response_model=Product)
def get_product(sku: str, conn: Connection = Depends(get_connection)):
    row = conn.execute(text(f"{_BASE_QUERY} WHERE sku = :sku"), {"sku": sku}).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="product not found")
    return row
