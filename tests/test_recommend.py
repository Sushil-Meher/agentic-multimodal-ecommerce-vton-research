from fastapi.testclient import TestClient

from src.api.db import engine
from src.api.main import app
from src.search.recommend import rank_products, recommend_products

client = TestClient(app)


def test_ranking_orders_by_number_of_matched_preferences():
    candidates = [
        {"sku": "LOW", "primary_color": "Blue", "occasion": "Casual", "bandha_type": None, "fabric": None, "thread_count": None},
        {"sku": "HIGH", "primary_color": "Maroon", "occasion": "Wedding", "bandha_type": "Traditional Bandha", "fabric": "Cotton", "thread_count": 120},
        {"sku": "MID", "primary_color": "Maroon", "occasion": "Wedding", "bandha_type": None, "fabric": None, "thread_count": None},
    ]

    ranked = rank_products(
        candidates,
        max_price=None,
        color="Maroon",
        bandha_type="Traditional Bandha",
        thread_count=120,
        occasion="Wedding",
        fabric="Cotton",
    )

    assert [c["sku"] for c in ranked] == ["HIGH", "MID", "LOW"]
    assert ranked[0]["score"] > ranked[1]["score"] > ranked[2]["score"]
    assert "wedding" in ranked[0]["reason"].lower()


def test_unavailable_products_excluded(product_factory):
    available = product_factory(price=1000, inventory_status="AVAILABLE", occasion="Wedding")
    sold_out = product_factory(price=1000, inventory_status="SOLD", occasion="Wedding")

    with engine.connect() as conn:
        results = recommend_products(conn, occasion="Wedding")

    skus = [r["sku"] for r in results]
    assert available in skus
    assert sold_out not in skus


def test_budget_constraint_excludes_expensive_products(product_factory):
    cheap = product_factory(price=3000, inventory_status="AVAILABLE")
    pricey = product_factory(price=9000, inventory_status="AVAILABLE")

    with engine.connect() as conn:
        results = recommend_products(conn, max_price=5000)

    skus = [r["sku"] for r in results]
    assert cheap in skus
    assert pricey not in skus


def test_multiple_preferences_affect_ranking(product_factory):
    best_match = product_factory(
        price=1000,
        inventory_status="AVAILABLE",
        primary_color="Maroon",
        occasion="Wedding",
        bandha_type="Traditional Bandha",
    )
    partial_match = product_factory(
        price=1000, inventory_status="AVAILABLE", primary_color="Maroon", occasion="Casual"
    )

    with engine.connect() as conn:
        results = recommend_products(
            conn, color="Maroon", occasion="Wedding", bandha_type="Traditional Bandha"
        )

    skus = [r["sku"] for r in results]
    assert skus.index(best_match) < skus.index(partial_match)


def test_no_candidates_returns_empty_list(product_factory):
    # color/occasion/etc. are soft preferences (used for scoring, not
    # filtering), so a mismatched color still comes back with a low score.
    # only a hard constraint like budget can produce zero candidates.
    product_factory(price=1000, inventory_status="AVAILABLE", primary_color="Blue")

    with engine.connect() as conn:
        results = recommend_products(conn, max_price=1)

    assert results == []


def test_unmatched_soft_preferences_still_return_low_scored_results(product_factory):
    sku = product_factory(price=1000, inventory_status="AVAILABLE", primary_color="Blue")

    with engine.connect() as conn:
        results = recommend_products(conn, color="no-such-color-xyz")

    match = next(r for r in results if r["sku"] == sku)
    assert match["score"] == 0
    assert match["reason"] == "matches your search"


def test_include_price_false_excludes_price_field(product_factory):
    sku = product_factory(price=1000, inventory_status="AVAILABLE")

    with engine.connect() as conn:
        results = recommend_products(conn, include_price=False)

    result = next(r for r in results if r["sku"] == sku)
    assert "price" not in result


def test_recommend_eval_fixture_excluded_by_default(product_factory):
    sku = product_factory(
        status="ACTIVE", price=1000, inventory_status="AVAILABLE",
        sku="EVAL-TEST-0003", primary_color="Maroon",
    )

    with engine.connect() as conn:
        results = recommend_products(conn, color="Maroon")

    assert sku not in [r["sku"] for r in results]


def test_recommend_eval_fixture_included_when_requested(product_factory):
    sku = product_factory(
        status="ACTIVE", price=1000, inventory_status="AVAILABLE",
        sku="EVAL-TEST-0004", primary_color="Maroon",
    )

    with engine.connect() as conn:
        results = recommend_products(conn, color="Maroon", include_eval_fixtures=True)

    assert sku in [r["sku"] for r in results]


def test_recommendations_endpoint_hides_price(product_factory):
    sku = product_factory(
        status="ACTIVE", price=4321, inventory_status="AVAILABLE", occasion="Wedding"
    )
    resp = client.get("/products/recommendations", params={"occasion": "Wedding"})
    assert resp.status_code == 200
    match = next(p for p in resp.json() if p["sku"] == sku)
    assert "price" not in match
    assert "score" in match
    assert "reason" in match
