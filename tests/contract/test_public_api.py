import spin_rag as pkg


def test_public_exports():
    for name in pkg.__all__:
        assert hasattr(pkg, name), name


def test_version():
    assert pkg.__version__ == "0.2.0"


def test_demo_imports_resolve():
    from spin_rag import (
        BACKEND_LLAMACPP,
        BACKEND_OPENROUTER,
        DEFAULT_EMBED_MODEL,
        DEFAULT_LLAMACPP_BASE_URL,
        DEFAULT_LLAMACPP_MODEL,
        DEFAULT_LLM_MODEL,
        DEFAULT_OPENROUTER_BASE_URL,
        SpinRAG,
    )

    assert BACKEND_LLAMACPP == "llamacpp"
    assert BACKEND_OPENROUTER == "openrouter"
    assert DEFAULT_LLAMACPP_BASE_URL.endswith("/v1")
    assert DEFAULT_LLAMACPP_MODEL
    assert DEFAULT_LLM_MODEL
    assert DEFAULT_EMBED_MODEL
    assert DEFAULT_OPENROUTER_BASE_URL
    assert SpinRAG is not None
