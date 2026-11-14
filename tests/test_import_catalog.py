import json
import uuid
from array import array

import pytest
from sqlalchemy import text

from src.api.db import engine
from src.retrieval import import_catalog as ic

REC = {
    "observable": {
        "body_colour": "deep green", "border_colour": "coffee", "pallu_colour": "not_sure",
        "body_other_colours": ["gold"], "blouse": "blouse_work",
        "body_motifs": {"animals_birds": ["bird"], "flowers_plants": ["lotus"], "geometric": [], "other": []},
        "pallu_motifs": {"animals_birds": ["elephant"], "flowers_plants": [], "geometric": [], "other": []},
    },
    "traditional_classification": {
        "body_pattern": {"human_label": "box pattern", "status": "human_labelled"},
        "border_class": {"human_label": None, "status": "not_assessed"},
        "pallu_class": {"human_label": "sample pallu", "status": "human_labelled"},
    },
}
PHOTOS = [{"image_id": "T-1", "image_path": "t/a.jpg", "image_type": "flat_lay"},
          {"image_id": "T-2", "image_path": "t/b.jpg", "image_type": "worn"}]
VECS = [[0.1] * 768, [0.2] * 768]


def test_record_to_rows_maps_labels():
    cols, attrs = ic.record_to_rows(REC)
    assert cols == {"primary_color": "deep green", "secondary_color": "coffee", "border_type": None, "blouse_included": True}
    assert ("colour", "deep green", "body", None) in attrs
    assert ("colour", "not_sure", "pallu", None) not in attrs
    assert ("motif", "elephant", "pallu", "animals_birds") in attrs
    assert ("label", "sample pallu", "pallu", None) in attrs
    assert ("blouse", "blouse_work", "blouse", None) in attrs
    assert not any(a[0] == "label" and a[2] == "border" for a in attrs)


def test_no_blouse_means_not_included():
    rec = json.loads(json.dumps(REC))
    rec["observable"]["blouse"] = "no_blouse"
    assert ic.record_to_rows(rec)[0]["blouse_included"] is False


def test_read_npy(tmp_path):
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }"
    header += " " * (63 - len(header)) + "\n"
    data = array("f", [1, 2, 3, 4, 5, 6])
    path = tmp_path / "v.npy"
    path.write_bytes(b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header.encode() + data.tobytes())
    assert ic.read_npy(path) == [[1, 2, 3], [4, 5, 6]]


@pytest.fixture
def sku():
    value = f"TEST-IMPORT-{uuid.uuid4().hex[:8].upper()}"
    yield value
    with engine.begin() as conn:
        pid = conn.execute(text("SELECT id FROM catalog.products WHERE sku = :s"), {"s": value}).scalar_one_or_none()
        if pid:
            conn.execute(text("DELETE FROM commerce.product_prices WHERE product_id = :p"), {"p": pid})
            conn.execute(text("DELETE FROM commerce.inventory_items WHERE product_id = :p"), {"p": pid})
            conn.execute(text("DELETE FROM catalog.products WHERE id = :p"), {"p": pid})


def counts(conn, pid):
    return [conn.execute(text(q), {"p": pid}).scalar_one() for q in (
        "SELECT count(*) FROM catalog.product_attributes WHERE product_id = :p",
        "SELECT count(*) FROM catalog.product_images WHERE product_id = :p",
        "SELECT count(*) FROM catalog.product_embeddings WHERE product_id = :p")]


def test_import_is_idempotent_and_draft(sku):
    with engine.begin() as conn:
        pid = ic.import_saree(conn, sku, REC, PHOTOS, VECS, "test-model", {"snapshot": "x"})
        first = counts(conn, pid)
        ic.import_saree(conn, sku, REC, PHOTOS, VECS, "test-model", {"snapshot": "x"})
        assert counts(conn, pid) == first == [9, 2, 2]
        row = conn.execute(text("SELECT status, primary_color, blouse_included FROM catalog.products WHERE id = :p"), {"p": pid}).one()
        assert (row.status, row.primary_color, row.blouse_included) == ("DRAFT", "deep green", True)
        primary = conn.execute(text("SELECT image_type FROM catalog.product_images WHERE product_id = :p AND is_primary"), {"p": pid}).scalars().all()
        assert primary == ["flat_lay"]
        public = conn.execute(text("SELECT count(*) FROM catalog.public_product_attributes WHERE product_id = :p"), {"p": pid}).scalar_one()
        assert public == 0  # DRAFT products stay out of the public view


def test_prices_and_activation(sku):
    with engine.begin() as conn:
        pid = ic.import_saree(conn, sku, REC, PHOTOS, VECS, "test-model", {})
        ic.apply_price(conn, sku, 5500, "AVAILABLE")
        ic.apply_price(conn, sku, 5500, "AVAILABLE")
        ic.apply_price(conn, sku, 6100, "AVAILABLE")
        prices = conn.execute(text("SELECT price, is_current FROM commerce.product_prices WHERE product_id = :p ORDER BY created_at"), {"p": pid}).all()
        assert [(float(p), c) for p, c in prices] == [(5500.0, False), (6100.0, True)]
        assert conn.execute(text("SELECT status FROM catalog.products WHERE id = :p"), {"p": pid}).scalar_one() == "DRAFT"
        ic.apply_price(conn, sku, 6100, "AVAILABLE", activate=True)
        assert conn.execute(text("SELECT status FROM catalog.products WHERE id = :p"), {"p": pid}).scalar_one() == "ACTIVE"
        assert conn.execute(text("SELECT count(*) FROM catalog.public_product_attributes WHERE product_id = :p"), {"p": pid}).scalar_one() == 9


def test_price_sheet_validation(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("sku,price_inr,stock_status,notes\nA,\"5,500\",,\nB,,,\nC,6000,sold,\n", encoding="utf-8")
    assert ic.parse_price_rows(path) == [("A", 5500.0, "AVAILABLE"), ("C", 6000.0, "SOLD")]
    for bad in ("A,abc,,\n", "A,-5,,\n", "A,100,gone,\n"):
        path.write_text("sku,price_inr,stock_status,notes\n" + bad, encoding="utf-8")
        with pytest.raises(ValueError):
            ic.parse_price_rows(path)


def test_display_name():
    assert ic.display_name("SAM-DATA-0007") == "BunkarHouse00007"
    assert ic.display_name("SAM-DATA-0126") == "BunkarHouse00126"
    assert ic.display_name("OTHER-1") == "Sambalpuri saree OTHER-1 (draft)"


def test_hand_edited_name_survives_reimport(sku):
    with engine.begin() as conn:
        pid = ic.import_saree(conn, sku, REC, PHOTOS, VECS, "test-model", {})
        conn.execute(text("UPDATE catalog.products SET name = 'My own name' WHERE id = :p"), {"p": pid})
        ic.import_saree(conn, sku, REC, PHOTOS, VECS, "test-model", {})
        assert conn.execute(text("SELECT name FROM catalog.products WHERE id = :p"), {"p": pid}).scalar_one() == "My own name"
