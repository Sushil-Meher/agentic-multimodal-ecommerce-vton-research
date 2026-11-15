import json

from src.agent import graph


def test_glossary_is_added_to_the_prompt_when_the_file_exists(tmp_path, monkeypatch):
    path = tmp_path / "glossary.json"
    path.write_text(json.dumps({"terms": [
        {"term": "teracotta", "aliases": ["terracotta"], "meaning": "a body pattern, search it as a design, never a colour"},
        {"term": "milan", "meaning": "small white squares"},
    ]}), encoding="utf-8")
    monkeypatch.setenv("AGENT_GLOSSARY_FILE", str(path))
    text = graph.load_glossary()
    assert '"teracotta" (also written terracotta): a body pattern' in text
    assert '"milan": small white squares' in text


def test_missing_or_broken_glossary_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_GLOSSARY_FILE", str(tmp_path / "nope.json"))
    assert graph.load_glossary() == ""
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("AGENT_GLOSSARY_FILE", str(bad))
    assert graph.load_glossary() == ""
    bad.write_text(json.dumps({"other": 1}), encoding="utf-8")
    assert graph.load_glossary() == ""


def test_prompt_tells_the_agent_not_to_claim_none_exist_and_to_use_names():
    assert "never say none exist" in graph.SYSTEM_PROMPT
    assert "never by SKU" in graph.SYSTEM_PROMPT


def test_prompt_requires_searching_before_saying_something_is_unknown():
    assert "Always run a search before" in graph.SYSTEM_PROMPT
