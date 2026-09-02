import uuid

import pytest
from sqlalchemy import text

from src.api.db import engine


@pytest.fixture
def product_factory():
    """Creates throwaway products (+ optional price/inventory) directly in
    the DB and cleans them up after. Keeps the real sample data untouched."""
    created_ids = []

    def make(status="ACTIVE", price=None, inventory_status=None, **overrides):
        sku = overrides.get("sku") or f"TEST-{uuid.uuid4().hex[:10].upper()}"
        slug = overrides.get("public_slug") or sku.lower()
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    INSERT INTO catalog.products
                        (sku, name, status, public_slug, primary_color, bandha_type,
                         thread_count, occasion, fabric)
                    VALUES
                        (:sku, :name, :status, :slug, :color, :bandha_type,
                         :thread_count, :occasion, :fabric)
                    RETURNING id
                    """
                ),
                {
                    "sku": sku,
                    "name": overrides.get("name", "Test Saree"),
                    "status": status,
                    "slug": slug,
                    "color": overrides.get("primary_color"),
                    "bandha_type": overrides.get("bandha_type"),
                    "thread_count": overrides.get("thread_count"),
                    "occasion": overrides.get("occasion"),
                    "fabric": overrides.get("fabric"),
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

            if inventory_status is not None:
                conn.execute(
                    text(
                        """
                        INSERT INTO commerce.inventory_items (product_id, inventory_code, status)
                        VALUES (:pid, :code, :status)
                        """
                    ),
                    {"pid": product_id, "code": f"{sku}-INV", "status": inventory_status},
                )

        created_ids.append(product_id)
        return sku

    yield make

    with engine.begin() as conn:
        for product_id in created_ids:
            conn.execute(
                text("DELETE FROM commerce.inventory_items WHERE product_id = :id"),
                {"id": product_id},
            )
            conn.execute(
                text("DELETE FROM commerce.product_prices WHERE product_id = :id"),
                {"id": product_id},
            )
            conn.execute(text("DELETE FROM catalog.products WHERE id = :id"), {"id": product_id})
