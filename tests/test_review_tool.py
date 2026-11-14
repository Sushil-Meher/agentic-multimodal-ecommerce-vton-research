import csv
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "review_server", Path(__file__).parents[1] / "tools" / "review_tool" / "server.py"
)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


@pytest.fixture
def store(tmp_path):
    meta = tmp_path / "metadata"
    meta.mkdir()
    rows = [
        {"image_id": "F01-I01", "saree_id": "SAM-DATA-0001", "image_path": "1/a.jpg", "image_type": "flat_lay", "width": "10", "height": "10"},
        {"image_id": "F01-I02", "saree_id": "SAM-DATA-0001", "image_path": "1/b.jpg", "image_type": "worn", "width": "10", "height": "10"},
        {"image_id": "F00-I01", "saree_id": "UNRESOLVED", "image_path": "x/c.jpg", "image_type": "worn", "width": "10", "height": "10"},
    ]
    with open(meta / "image_annotations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    return server.Store(tmp_path, tmp_path / "photos")


def payload(**over):
    base = {"status": "verified",
            "observable": {"body_colour": "yellow", "border_colour": "maroon", "pallu_colour": "not_sure",
                           "body_other_colours": ["gold"], "body_motifs": {"animals_birds": ["elephant"]},
                           "pallu_motifs": {"animals_birds": ["peacock"]}},
            "image_type_corrections": {"F01-I02": "blouse_work"}, "classification": {"border_class": ""}, "notes": " ok "}
    base.update(over)
    return base


def test_unresolved_images_are_not_listed(store):
    assert store.saree_ids() == ["SAM-DATA-0001"]


def test_save_and_reload(store):
    store.save("SAM-DATA-0001", payload())
    rec = store.record("SAM-DATA-0001")
    assert rec["observable"]["border_colour"] == "maroon"
    assert rec["observable"]["pallu_colour"] == "not_sure"
    assert rec["notes"] == "ok"
    assert rec["traditional_classification"]["border_class"]["status"] == "not_assessed"
    types = {i["image_id"]: i["image_type"] for i in store.image_list("SAM-DATA-0001")}
    assert types["F01-I02"] == "blouse_work"
    assert store.summary()[0]["status"] == "verified"


def test_resave_keeps_history(store):
    store.save("SAM-DATA-0001", payload())
    store.save("SAM-DATA-0001", payload(notes="second"))
    assert len(list(store.history_dir.glob("SAM-DATA-0001_*.json"))) == 1


@pytest.mark.parametrize("bad", [
    {"status": "done"},
    {"observable": {"blouse": "silk"}},
    {"observable": {"body_colour": "x" * 50}},
    {"observable": {"body_other_colours": [""]}},
    {"image_type_corrections": {"F99-I01": "worn"}},
    {"image_type_corrections": {"F01-I01": "portrait"}},
    {"notes": "x" * 3000},
])
def test_invalid_payloads_rejected(store, bad):
    with pytest.raises(ValueError):
        store.save("SAM-DATA-0001", payload(**bad))
    assert store.record("SAM-DATA-0001") is None


def test_unknown_saree_rejected(store):
    with pytest.raises(ValueError):
        store.save("SAM-DATA-0099", payload())


def test_vocabulary_seeded_and_extended(store):
    assert "elephant" in store.vocab()["animals_birds"]
    store.add_term("animals_birds", "parrot")
    store.add_term("animals_birds", "parrot")
    assert store.vocab()["animals_birds"].count("parrot") == 1
    with pytest.raises(ValueError):
        store.add_term("bogus", "x")
    assert json.loads(store.vocab_path.read_text(encoding="utf-8"))["animals_birds"][-1] == "parrot"


def test_labels_saved_and_offered_as_suggestions(store):
    store.save("SAM-DATA-0001", payload(classification={"border_class": "  my border  ", "pallu_class": "my pallu", "body_pattern": ""}))
    rec = store.record("SAM-DATA-0001")["traditional_classification"]
    assert rec["border_class"] == {"human_label": "my border", "status": "human_labelled"}
    assert rec["body_pattern"]["status"] == "not_assessed"
    opts = store.label_options()
    assert opts["border_class"] == ["my border"] and opts["pallu_class"] == ["my pallu"] and opts["body_pattern"] == []
    assert set(opts) == set(server.CLASS_FIELDS)


def test_custom_colours_accepted_and_remembered(store):
    assert "deep green" in store.vocab()["colours"]
    store.save("SAM-DATA-0001", payload(observable={"body_colour": " Moss Green ", "border_colour": "not sure",
                                                    "pallu_colour": "maroon", "body_other_colours": ["Rust"]}))
    obs = store.record("SAM-DATA-0001")["observable"]
    assert obs["body_colour"] == "moss green" and obs["border_colour"] == "not_sure" and obs["body_other_colours"] == ["rust"]
    colours = store.vocab()["colours"]
    assert "moss green" in colours and "rust" in colours and "not_sure" not in colours


def test_blouse_option_saved(store):
    store.save("SAM-DATA-0001", payload(observable={"blouse": "blouse_work"}))
    assert store.record("SAM-DATA-0001")["observable"]["blouse"] == "blouse_work"
    store.save("SAM-DATA-0001", payload(observable={"blouse": "no_blouse"}))
    assert store.record("SAM-DATA-0001")["observable"]["blouse"] == "no_blouse"
