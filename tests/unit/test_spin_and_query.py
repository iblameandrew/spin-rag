from spin_rag import BACKEND_LLAMACPP, SpinRAG, SpinType
from tests.fakes import FakeBackend


def _rag(backend: FakeBackend, content: str, n_epochs: int = 0) -> SpinRAG:
    return SpinRAG(
        content=content,
        n_epochs=n_epochs,
        backend=BACKEND_LLAMACPP,
        backend_instance=backend,
    )


def test_empty_content_skips_index(fake_backend):
    rag = _rag(fake_backend, "")
    assert rag.documents == []
    assert rag.query("anything") == "No TOP spin documents available for querying."


def test_empty_query_returns_empty(fake_backend):
    rag = _rag(fake_backend, "Revenue 2026-Q1-006 | Client: Globex")
    assert rag.query("   ") == ""


def test_top_query_is_verbatim(fake_backend, corpus):
    rag = _rag(fake_backend, corpus, n_epochs=0)
    tops = [d for d in rag.documents if d.spin == SpinType.TOP]
    assert tops
    answer = rag.query("Globex Industries Q1 revenue")
    assert answer in {d.text for d in tops}


def test_left_query_calls_chat(fake_backend, corpus):
    rag = _rag(fake_backend, corpus, n_epochs=0)
    fake_backend.chat_prompts.clear()
    answer = rag.query("Acme Corp. -")
    assert answer == fake_backend.fusion_text
    assert any("Incomplete query" in p for p in fake_backend.chat_prompts)


def test_right_query_calls_chat(fake_backend, corpus):
    rag = _rag(fake_backend, corpus, n_epochs=0)
    fake_backend.chat_prompts.clear()
    answer = rag.query("Vendor: Initech | Period: 2026 | Type: revenue")
    assert answer == fake_backend.fusion_text
    assert any("Definition/parameters" in p for p in fake_backend.chat_prompts)


def test_bottom_query_falls_back_to_top_when_no_neighbors(fake_backend, corpus):
    rag = _rag(fake_backend, corpus, n_epochs=0)
    # n_epochs=0 means no fusion edges; BOTTOM walks adjacency and falls back.
    answer = rag.query("What revenue did Acme Corp. generate in 2026-Q1?")
    tops = {d.text for d in rag.documents if d.spin == SpinType.TOP}
    assert answer in tops


def test_reorganize_graph_adds_nodes(fake_backend, corpus):
    rag = _rag(fake_backend, corpus, n_epochs=0)
    before = len(rag.documents)
    rag.query("Globex Industries Q1 revenue", reorganize_graph=True)
    assert len(rag.documents) > before
    labels = {e["label"] for e in rag.graph["edges"]}
    assert "queries" in labels
    assert "generates_response" in labels


def test_get_verbose_log_and_clear(fake_backend):
    rag = _rag(fake_backend, "Revenue line")
    assert rag.get_verbose_log()
    rag.clear_log()
    assert rag.get_verbose_log() == []


def test_embedding_failure_falls_back_to_zero(corpus):
    backend = FakeBackend(
        spin_by_substring={"Acme": "LEFT"},
        embed_error=RuntimeError("down"),
    )
    rag = _rag(backend, "Revenue complete row\nAcme Corp. -", n_epochs=0)
    assert rag.query("Revenue")  # still works via zero vectors


def test_chat_failure_during_spin_defaults_top():
    backend = FakeBackend(chat_error=RuntimeError("boom"))
    rag = SpinRAG(
        content="mystery line",
        n_epochs=0,
        backend=BACKEND_LLAMACPP,
        backend_instance=backend,
    )
    assert all(d.spin == SpinType.TOP for d in rag.documents)
