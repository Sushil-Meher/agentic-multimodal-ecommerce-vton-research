import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from src.api.db import engine
from src.api.main import app

client = TestClient(app)


@pytest.fixture
def product_factory():
    """Creates throwaway test products directly in the DB and cleans them up after."""
    created_ids = []

    def make(status="ACTIVE", price=None, **overrides):
        sku = overrides.get("sku") or f"TEST-{uuid.uuid4().hex[:10].upper()}"
        slug = overrides.get("public_slug") or sku.lower()
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    INSERT INTO catalog.products
                        (sku, name, status, public_slug, primary_color, thread_count)
                    VALUES
                        (:sku, :name, :status, :slug, :color, :thread_count)
                    RETURNING id
                    """
                ),
                {
                    "sku": sku,
                    "name": overrides.get("name", "Test Saree"),
                    "status": status,
                    "slug": slug,
                    "color": overrides.get("primary_color"),
                    "thread_count": overrides.get("thread_count"),
                },
            ).one()
            product_id = row.id
            if price is not None:
                conn.execute(
                    text(
                        "INSERT INTO commerce.product_prices (product_id, price) VALUES (:pid, :price)"
                    ),
                    {"pid": product_id, "price": price},
                )
        created_ids.append(product_id)
        return sku

    yield make

    with engine.begin() as conn:
        for product_id in created_ids:
            conn.execute(
                text("DELETE FROM commerce.product_prices WHERE product_id = :id"),
                {"id": product_id},
            )
            conn.execute(text("DELETE FROM catalog.products WHERE id = :id"), {"id": product_id})


def test_public_endpoint_hides_price(product_factory):
    sku = product_factory(status="ACTIVE", price=1234, primary_color="Teal")
    resp = client.get(f"/products/{sku}")
    assert resp.status_code == 200
    assert "price" not in resp.json()


def test_active_product_is_listed(product_factory):
    sku = product_factory(status="ACTIVE", primary_color="Teal")
    resp = client.get("/products")
    assert resp.status_code == 200
    assert sku in [p["sku"] for p in resp.json()]


def test_draft_product_not_returned_publicly(product_factory):
    sku = product_factory(status="DRAFT")

    list_resp = client.get("/products")
    assert sku not in [p["sku"] for p in list_resp.json()]

    detail_resp = client.get(f"/products/{sku}")
    assert detail_resp.status_code == 404


def test_get_product_not_found():
    resp = client.get("/products/DOES-NOT-EXIST")
    assert resp.status_code == 404


def test_search_by_color(product_factory):
    sku = product_factory(status="ACTIVE", primary_color="Testmagenta")
    resp = client.get("/products/search", params={"color": "testmagenta"})
    assert resp.status_code == 200
    assert sku in [p["sku"] for p in resp.json()]


def test_search_by_max_thread_count(product_factory):
    low_sku = product_factory(status="ACTIVE", thread_count=40)
    high_sku = product_factory(status="ACTIVE", thread_count=200)
    resp = client.get("/products/search", params={"max_thread_count": 50})
    skus = [p["sku"] for p in resp.json()]
    assert low_sku in skus
    assert high_sku not in skus


def test_search_rejects_invalid_thread_count():
    resp = client.get("/products/search", params={"max_thread_count": 0})
    assert resp.status_code == 422
