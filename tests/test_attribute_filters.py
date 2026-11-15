import json

import pytest
from sqlalchemy import text

from src.agent.tools import search_by_image, search_products
from src.api.db import engine
from src.retrieval import search as photo_search
from src.search.attributes import colour_match, escape_like, normalise_blouse, parse_terms
from src.search.products import search_products as search_service


def make(product_factory, *, border=None, attrs=(), vector=None, **kwargs):
    """Product with attributes (attribute, value, region), optional border colour and one photo vector."""
    kwargs.setdefault("price", 8000)
    kwargs.setdefault("inventory_status", "AVAILABLE")
    sku = product_factory(**kwargs)
    with engine.begin() as conn:
        pid = conn.execute(text("SELECT id FROM catalog.products WHERE sku = :s"), {"s": sku}).scalar_one()
        if border:
            conn.execute(text("UPDATE catalog.products SET secondary_color = :c WHERE id = :p"), {"c": border, "p": pid})
        for attribute, value, region in attrs:
            conn.execute(text("INSERT INTO catalog.product_attributes (product_id, attribute, value, region) VALUES (:p, :a, :v, :r)"),
                         {"p": pid, "a": attribute, "v": value, "r": region})
        if vector:
            conn.execute(text("""INSERT INTO catalog.product_embeddings (product_id, embedding_type, model_name, embedding)
                                 VALUES (:p, 'photo_whole', :m, CAST(:v AS vector))"""),
                         {"p": pid, "m": photo_search.MODEL_NAME, "v": photo_search.vector_literal(vector)})
    return sku


def skus(**filters):
    with engine.connect() as conn:
        return [r["sku"] for r in search_service(conn, **filters)]


def test_parse_terms():
    assert parse_terms("Red and Black") == ["red", "black"]
    assert parse_terms("red, black & gold + maroon / pink") == ["red", "black", "gold"]
    assert parse_terms("  ") == [] and parse_terms(None) == []
    assert parse_terms("red and red") == ["red"]
    assert parse_terms("sand") == ["sand"]  # 'and' inside a word is not a separator


def test_escape_like_and_blouse_normalising():
    assert escape_like("50%_off\\") == "50\\%\\_off\\\\"
    assert normalise_blouse("Blouse Work") == "blouse_work" and normalise_blouse("no blouse") == "no_blouse"
    with pytest.raises(ValueError):
        normalise_blouse("silk")


def test_colour_rule_single_colour_needs_main_body_colour():
    assert colour_match("red", "black", [], ["red"])[0] == "best"
    assert colour_match("light red", None, [], ["red"])[0] == "best"
    assert colour_match("green", "red", ["red"], ["red"]) is None


def test_colour_rule_two_colours_tiers():
    assert colour_match("red", "black", [], ["red", "black"])[0] == "best"      # red body + black border
    assert colour_match("black", "red", [], ["red", "black"])[0] == "best"      # black body + red border
    assert colour_match("red", "gold", ["black"], ["red", "black"])[0] == "best"  # both in body, red main
    tier, note = colour_match("green", "red", ["black"], ["red", "black"])
    assert tier == "weaker" and "main body colour is green" in note
    assert colour_match("red", "gold", [], ["red", "black"]) is None            # black missing
    assert colour_match("red black", None, [], ["red", "black"])[0] == "best"  # both in the main body colour
    assert colour_match("yellow", "maroon", ["maroon"], ["yellow", "maroon"])[1].endswith("maroon in the border")


def test_blouse_filter(product_factory):
    work = make(product_factory, attrs=[("blouse", "blouse_work", "blouse")])
    plain = make(product_factory, attrs=[("blouse", "plain", "blouse")])
    found = skus(blouse="blouse work")
    assert work in found and plain not in found


def test_design_matches_label_or_motif_by_contains(product_factory):
    label = make(product_factory, attrs=[("label", "star milan", "body")])
    motif = make(product_factory, attrs=[("motif", "Milan box", "body")])
    other = make(product_factory, attrs=[("label", "bandha paper", "body")])
    found = skus(design="MILAN")
    assert label in found and motif in found and other not in found


def test_design_region_and_several_words(product_factory):
    both = make(product_factory, attrs=[("label", "milan", "body"), ("motif", "peacock", "pallu")])
    wrong_region = make(product_factory, attrs=[("label", "milan", "border"), ("motif", "peacock", "pallu")])
    assert both in skus(design="milan and peacock")
    assert both in skus(design="milan", region="body")
    assert wrong_region not in skus(design="milan", region="body")
    assert wrong_region in skus(design="milan", region="border")
    assert both not in skus(design="milan and elephant")


def test_design_wildcards_are_literal(product_factory):
    sku = make(product_factory, attrs=[("label", "milan", "body")])
    assert sku not in skus(design="%")
    assert sku not in skus(design="m_lan")


def test_colour_search_orders_best_first_and_labels_results(product_factory):
    weaker = make(product_factory, primary_color="green", border="red", attrs=[("colour_other", "black", "body")])
    best = make(product_factory, primary_color="red", border="black")
    miss = make(product_factory, primary_color="red", border="gold")
    with engine.connect() as conn:
        rows = [r for r in search_service(conn, colors="red and black") if r["sku"] in (weaker, best, miss)]
    assert [r["sku"] for r in rows] == [best, weaker]
    assert [r["colour_match"] for r in rows] == ["best", "weaker"]
    assert "body_other_colours" not in rows[0]


def test_filters_combine_with_price_range(product_factory):
    cheap = make(product_factory, price=6000, primary_color="red", attrs=[("blouse", "blouse_work", "blouse")])
    dear = make(product_factory, price=12000, primary_color="red", attrs=[("blouse", "blouse_work", "blouse")])
    found = skus(colors="red", blouse="blouse work", min_price=5000, max_price=9000)
    assert cheap in found and dear not in found


def test_draft_products_hidden_unless_development_flag(product_factory):
    draft = make(product_factory, status="DRAFT", attrs=[("label", "draftonly", "body")])
    assert draft not in skus(design="draftonly")
    assert draft in skus(design="draftonly", include_draft=True)


def test_invalid_region_or_blouse_rejected(product_factory):
    with engine.connect() as conn:
        with pytest.raises(ValueError):
            search_service(conn, region="sleeve")
        with pytest.raises(ValueError):
            search_service(conn, blouse="silk")


def test_agent_tool_accepts_new_filters_and_placeholders(product_factory):
    sku = make(product_factory, primary_color="Teal", attrs=[("label", "toolword", "body")])
    payload = json.loads(search_products.invoke({"design": "toolword", "colors": "", "blouse": "", "min_price": 0}))
    assert sku in [p["sku"] for p in payload]
    assert "error" in json.loads(search_products.invoke({"region": "sleeve"}))
    # the older single-colour argument now goes through the same colour rule
    assert sku in [p["sku"] for p in json.loads(search_products.invoke({"color": "teal"}))]


def test_tool_is_immune_to_sql_injection_in_design():
    payload = json.loads(search_products.invoke({"design": "x'; DROP TABLE catalog.products; --"}))
    assert payload == []
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM catalog.products")).scalar_one() > 0


# ---- photo search

def unit(i, extra=0.0):
    v = [0.0] * 768
    v[i] = 1.0
    if extra:
        v[(i + 1) % 768] = extra
    return v


def test_photo_search_ranks_by_similarity_within_filters(product_factory, monkeypatch):
    near = make(product_factory, price=7000, vector=unit(0, 0.2), attrs=[("blouse", "plain", "blouse")])
    far = make(product_factory, price=7000, vector=unit(5), attrs=[("blouse", "plain", "blouse")])
    filtered_out = make(product_factory, price=15000, vector=unit(0))
    monkeypatch.setattr(photo_search, "embed_image", lambda path: unit(0))
    with engine.connect() as conn:
        rows = photo_search.search_by_image(conn, "q.jpg", limit=50, max_price=9000)
    order = [r["sku"] for r in rows if r["sku"] in (near, far, filtered_out)]
    assert order == [near, far]
    assert rows[0]["similarity"] >= rows[-1]["similarity"]


def test_upload_folder_is_enforced(tmp_path, monkeypatch):
    folder = tmp_path / "uploads"
    folder.mkdir()
    (folder / "ok.jpg").write_bytes(b"x")
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"x")
    monkeypatch.setattr(photo_search, "UPLOAD_DIR", str(folder.resolve()))
    assert photo_search.resolve_upload("ok.jpg").endswith("ok.jpg")
    for bad in (str(outside), "../secret.jpg", "missing.jpg"):
        with pytest.raises(ValueError):
            photo_search.resolve_upload(bad)


def test_photo_tool_returns_error_payload_instead_of_crashing():
    assert "error" in json.loads(search_by_image.invoke({"image_name": "no_such_photo.jpg"}))


def test_photo_tool_reports_service_down(tmp_path, monkeypatch):
    folder = tmp_path / "uploads"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(b"x")
    monkeypatch.setattr(photo_search, "UPLOAD_DIR", str(folder.resolve()))
    monkeypatch.setattr(photo_search, "SERVICE_URL", "http://127.0.0.1:9")  # nothing listens here
    result = json.loads(search_by_image.invoke({"image_name": "a.jpg"}))
    assert "not running" in result["error"]


def test_border_colour_filter(product_factory):
    red_border = make(product_factory, primary_color="green", border="red")
    gold_border = make(product_factory, primary_color="green", border="golden coffee")
    found = skus(border_color="red")
    assert red_border in found and gold_border not in found
    assert gold_border in skus(border_color="Coffee")  # contains, any case
    assert red_border in skus(colors="green", border_color="red")
    assert gold_border not in skus(colors="green", border_color="red")


def test_pallu_colour_filter(product_factory):
    maroon_pallu = make(product_factory, primary_color="pink", attrs=[("colour", "maroon", "pallu")])
    body_only = make(product_factory, primary_color="maroon", attrs=[("colour", "black", "pallu")])
    found = skus(pallu_color="maroon")
    assert maroon_pallu in found and body_only not in found  # a maroon BODY must not count as a maroon pallu


def test_border_and_pallu_filters_through_the_tool(product_factory):
    sku = make(product_factory, border="turquoise", attrs=[("colour", "plum", "pallu")])
    assert sku in [p["sku"] for p in json.loads(search_products.invoke({"border_color": "turquoise"}))]
    assert sku in [p["sku"] for p in json.loads(search_products.invoke({"pallu_color": "plum", "border_color": ""}))]


def test_colour_filters_are_injection_safe():
    for arg in ("border_color", "pallu_color"):
        assert json.loads(search_products.invoke({arg: "x'; DROP TABLE catalog.products; --"})) == []


@pytest.fixture
def families(tmp_path, monkeypatch):
    from src.search import attributes

    path = tmp_path / "families.json"
    path.write_text(json.dumps({"families": {"brown": ["brown", "coffee"]}}), encoding="utf-8")
    monkeypatch.setenv("COLOUR_FAMILIES_FILE", str(path))
    attributes._families.cache_clear()
    yield
    attributes._families.cache_clear()


def test_expand_colour_with_families(families):
    from src.search.attributes import expand_colour

    assert expand_colour("brown") == ["brown", "coffee"]
    assert expand_colour("deep brown") == ["deep brown", "brown", "coffee"]  # contains a family word
    assert "brown" in expand_colour("dark coffee")
    assert expand_colour("red") == ["red"]


def test_no_families_file_means_plain_matching():
    from src.search.attributes import expand_colour

    assert expand_colour("brown") == ["brown"]  # the autouse fixture points at a missing file


def test_similar_shade_matches_and_is_named_in_the_note(families):
    tier, note = colour_match("coffee", None, [], ["brown"])
    assert tier == "best" and note == "coffee (similar to brown) is the main body colour"
    assert colour_match("brown", None, [], ["brown"])[1] == "brown is the main body colour"
    assert colour_match("deep red", None, [], ["red"])[1] == "red is the main body colour"  # plain contains
    tier, note = colour_match("pink", "golden coffee", [], ["pink", "brown"])
    assert tier == "best" and note.endswith("golden coffee (similar to brown) in the border")


def test_brown_requests_find_coffee_sarees_exact_first(product_factory, families):
    coffee = make(product_factory, primary_color="coffee")
    brown = make(product_factory, primary_color="brown")
    deep = make(product_factory, primary_color="deep brown")
    other = make(product_factory, primary_color="blue")
    with engine.connect() as conn:
        rows = [r["sku"] for r in search_service(conn, colors="brown") if r["sku"] in (coffee, brown, deep, other)]
    assert set(rows) == {coffee, brown, deep}
    assert rows.index(coffee) > rows.index(brown) and rows.index(coffee) > rows.index(deep)  # literal brown first


def test_border_and_pallu_filters_use_families(product_factory, families):
    coffee_border = make(product_factory, border="golden coffee", attrs=[("colour", "coffee", "pallu")])
    red_border = make(product_factory, border="red")
    assert coffee_border in skus(border_color="brown") and red_border not in skus(border_color="brown")
    assert coffee_border in skus(pallu_color="deep brown")


@pytest.fixture
def glossary(tmp_path, monkeypatch):
    from src.search import attributes

    path = tmp_path / "glossary.json"
    path.write_text(json.dumps({"terms": [{"term": "teracotta", "aliases": ["terracotta", "Terakota"], "meaning": "x"},
                                           {"term": "milan", "meaning": "y"}]}), encoding="utf-8")
    monkeypatch.setenv("AGENT_GLOSSARY_FILE", str(path))
    attributes._glossary_groups.cache_clear()
    yield
    attributes._glossary_groups.cache_clear()


def test_design_spelling_variants_find_the_owners_spelling(product_factory, glossary):
    from src.search.attributes import expand_design

    assert expand_design("terracotta") == ["teracotta", "terracotta", "terakota"]
    assert expand_design("milan") == ["milan"] and expand_design("unlisted") == ["unlisted"]
    sku = make(product_factory, attrs=[("label", "teracotta", "body")])
    assert sku in skus(design="terracotta") and sku in skus(design="TERAKOTA") and sku in skus(design="teracotta")
    assert sku not in skus(design="terracotta and peacock")  # every word must still match


def test_without_a_glossary_spelling_must_match_literally(product_factory):
    sku = make(product_factory, attrs=[("label", "teracotta", "body")])
    assert sku not in skus(design="terracotta")


def test_tool_results_leave_out_ids_and_internal_fields(product_factory):
    sku = make(product_factory, primary_color="Teal", attrs=[("label", "slimcheck", "body")])
    row = next(p for p in json.loads(search_products.invoke({"design": "slimcheck"})) if p["sku"] == sku)
    for hidden in ("id", "public_slug", "created_at", "updated_at", "status", "description", "weight_grams"):
        assert hidden not in row
    assert row["name"] and row["price"] == 8000 and row["primary_color"] == "Teal" and row["blouse_included"] is False
