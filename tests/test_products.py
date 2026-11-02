from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


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
    sku = product_factory(
        status="ACTIVE", price=1000, inventory_status="AVAILABLE", primary_color="Testmagenta"
    )
    resp = client.get("/products/search", params={"color": "testmagenta"})
    assert resp.status_code == 200
    assert sku in [p["sku"] for p in resp.json()]


def test_search_by_thread_count(product_factory):
    match_sku = product_factory(
        status="ACTIVE", price=1000, inventory_status="AVAILABLE", thread_count=120
    )
    other_sku = product_factory(
        status="ACTIVE", price=1000, inventory_status="AVAILABLE", thread_count=80
    )
    resp = client.get("/products/search", params={"thread_count": 120})
    skus = [p["sku"] for p in resp.json()]
    assert match_sku in skus
    assert other_sku not in skus


def test_search_combines_multiple_filters(product_factory):
    match_sku = product_factory(
        status="ACTIVE",
        price=1000,
        inventory_status="AVAILABLE",
        primary_color="Crimson",
        occasion="Wedding",
    )
    wrong_occasion_sku = product_factory(
        status="ACTIVE",
        price=1000,
        inventory_status="AVAILABLE",
        primary_color="Crimson",
        occasion="Casual",
    )
    resp = client.get("/products/search", params={"color": "crimson", "occasion": "wedding"})
    skus = [p["sku"] for p in resp.json()]
    assert match_sku in skus
    assert wrong_occasion_sku not in skus


def test_search_excludes_out_of_stock_by_default(product_factory):
    sku = product_factory(status="ACTIVE", price=1000, inventory_status="SOLD")
    resp = client.get("/products/search")
    assert sku not in [p["sku"] for p in resp.json()]


def test_search_no_matches_returns_empty_list(product_factory):
    resp = client.get("/products/search", params={"color": "no-such-color-xyz"})
    assert resp.status_code == 200
    assert resp.json() == []


def test_search_rejects_invalid_thread_count():
    resp = client.get("/products/search", params={"thread_count": 0})
    assert resp.status_code == 422


def test_search_endpoint_hides_price(product_factory):
    sku = product_factory(status="ACTIVE", price=4321, inventory_status="AVAILABLE")
    resp = client.get("/products/search")
    match = next(p for p in resp.json() if p["sku"] == sku)
    assert "price" not in match


def test_eval_fixture_never_appears_in_public_endpoints(product_factory):
    sku = product_factory(
        status="ACTIVE", price=1000, inventory_status="AVAILABLE",
        sku="EVAL-TEST-0005", primary_color="Maroon", occasion="Wedding",
    )

    list_resp = client.get("/products")
    assert sku not in [p["sku"] for p in list_resp.json()]

    detail_resp = client.get(f"/products/{sku}")
    assert detail_resp.status_code == 404

    search_resp = client.get("/products/search", params={"color": "Maroon"})
    assert sku not in [p["sku"] for p in search_resp.json()]

    recommend_resp = client.get("/products/recommendations", params={"occasion": "Wedding"})
    assert sku not in [p["sku"] for p in recommend_resp.json()]
