import pytest

from src.api.db import engine
from src.search.products import search_products


def test_filters_by_price_range(product_factory):
    cheap = product_factory(price=1000, inventory_status="AVAILABLE")
    mid = product_factory(price=3000, inventory_status="AVAILABLE")
    pricey = product_factory(price=9000, inventory_status="AVAILABLE")

    with engine.connect() as conn:
        results = search_products(conn, min_price=2000, max_price=5000)

    skus = [r["sku"] for r in results]
    assert mid in skus
    assert cheap not in skus
    assert pricey not in skus


def test_filters_by_availability(product_factory):
    in_stock = product_factory(price=1000, inventory_status="AVAILABLE")
    sold_out = product_factory(price=1000, inventory_status="SOLD")

    with engine.connect() as conn:
        results = search_products(conn, in_stock_only=True)

    skus = [r["sku"] for r in results]
    assert in_stock in skus
    assert sold_out not in skus


def test_in_stock_only_false_includes_sold_out(product_factory):
    sold_out = product_factory(price=1000, inventory_status="SOLD")

    with engine.connect() as conn:
        results = search_products(conn, in_stock_only=False)

    assert sold_out in [r["sku"] for r in results]


def test_combines_color_and_occasion_filters(product_factory):
    match = product_factory(
        price=1000, inventory_status="AVAILABLE", primary_color="Emerald", occasion="Wedding"
    )
    wrong_occasion = product_factory(
        price=1000, inventory_status="AVAILABLE", primary_color="Emerald", occasion="Casual"
    )
    wrong_color = product_factory(
        price=1000, inventory_status="AVAILABLE", primary_color="Blue", occasion="Wedding"
    )

    with engine.connect() as conn:
        results = search_products(conn, color="emerald", occasion="wedding")

    skus = [r["sku"] for r in results]
    assert match in skus
    assert wrong_occasion not in skus
    assert wrong_color not in skus


def test_no_matching_products_returns_empty_list(product_factory):
    product_factory(price=1000, inventory_status="AVAILABLE", primary_color="Blue")

    with engine.connect() as conn:
        results = search_products(conn, color="no-such-color-xyz")

    assert results == []


def test_min_price_over_max_price_raises():
    with engine.connect() as conn:
        with pytest.raises(ValueError):
            search_products(conn, min_price=5000, max_price=1000)


def test_negative_max_price_raises():
    with engine.connect() as conn:
        with pytest.raises(ValueError):
            search_products(conn, max_price=-100)


def test_non_positive_thread_count_raises():
    with engine.connect() as conn:
        with pytest.raises(ValueError):
            search_products(conn, thread_count=0)


def test_include_price_false_excludes_price_field(product_factory):
    sku = product_factory(price=1000, inventory_status="AVAILABLE")

    with engine.connect() as conn:
        results = search_products(conn, include_price=False)

    result = next(r for r in results if r["sku"] == sku)
    assert "price" not in result


def test_include_price_true_returns_price_field(product_factory):
    sku = product_factory(price=1000, inventory_status="AVAILABLE")

    with engine.connect() as conn:
        results = search_products(conn, include_price=True)

    result = next(r for r in results if r["sku"] == sku)
    assert result["price"] == 1000
