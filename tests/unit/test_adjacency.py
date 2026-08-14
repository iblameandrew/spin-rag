from spin_rag import BACKEND_LLAMACPP, Document, SpinRAG, SpinType
from tests.fakes import FakeBackend


def test_bottom_query_uses_adjacent_right():
    backend = FakeBackend(
        spin_by_substring={"What revenue": "BOTTOM", "Vendor: Initech": "RIGHT"},
        default_spin="TOP",
        fusion_text="Crystallized Initech revenue",
    )
    rag = SpinRAG(
        content="Revenue 2026 | Client: Initech | Amount: $1\nVendor: Initech | Period: 2026 | Type: revenue",
        n_epochs=1,
        backend=BACKEND_LLAMACPP,
        backend_instance=backend,
    )
    backend.chat_prompts.clear()
    answer = rag.query("What revenue did Initech generate?")
    assert answer == "Crystallized Initech revenue"
    assert any("Evolutionary target" in p for p in backend.chat_prompts)


def test_get_adjacent_docs_both_directions():
    backend = FakeBackend()
    rag = SpinRAG(
        content="only",
        n_epochs=0,
        backend=BACKEND_LLAMACPP,
        backend_instance=backend,
    )
    other = Document(id="other", text="x", spin=SpinType.LEFT)
    rag.doc_map["other"] = other
    rag.graph["edges"].append({"source": rag.documents[0].id, "target": "other", "label": "x"})
    neighbors = rag._get_adjacent_docs(rag.documents[0].id)
    assert neighbors[0].id == "other"
