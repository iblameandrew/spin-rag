import pytest

from spin_rag.spin_rag import (
    BACKEND_LLAMACPP,
    BACKEND_OPENROUTER,
    DEFAULT_LLAMACPP_BASE_URL,
    _Backend,
)


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        _Backend(backend="ollama")


def test_openrouter_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        _Backend(backend=BACKEND_OPENROUTER)


def test_openrouter_headers(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://example.com")
    backend = _Backend(backend=BACKEND_OPENROUTER, app_name="SpinRAG-Test")
    assert backend.base_url.startswith("https://openrouter.ai")
    assert backend.headers["X-Title"] == "SpinRAG-Test"
    assert backend.headers["HTTP-Referer"] == "https://example.com"


def test_llamacpp_defaults_dummy_key(monkeypatch):
    monkeypatch.delenv("LLAMACPP_API_KEY", raising=False)
    backend = _Backend(backend=BACKEND_LLAMACPP)
    assert backend.base_url == DEFAULT_LLAMACPP_BASE_URL
    assert backend.api_key
    assert backend.headers == {}


def test_llamacpp_env_override(monkeypatch):
    monkeypatch.setenv("LLAMACPP_BASE_URL", "http://127.0.0.1:9090/v1")
    monkeypatch.setenv("LLAMACPP_API_KEY", "sk-local")
    backend = _Backend(backend=BACKEND_LLAMACPP)
    assert backend.base_url == "http://127.0.0.1:9090/v1"
    assert backend.api_key == "sk-local"
