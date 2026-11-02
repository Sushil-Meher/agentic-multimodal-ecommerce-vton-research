import pytest

from src.agent.llm import get_llm


def test_defaults_to_ollama_provider_and_baseline_model(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    llm = get_llm()

    assert llm.model == "qwen3:8b"


def test_ollama_uses_configured_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "qwen3:4b")

    llm = get_llm()

    assert llm.model == "qwen3:4b"


def test_ollama_uses_configured_base_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example.internal:11434")

    llm = get_llm()

    assert llm.base_url == "http://example.internal:11434"


def test_unsupported_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")

    with pytest.raises(ValueError):
        get_llm()
