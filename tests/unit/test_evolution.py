from spin_rag import BACKEND_LLAMACPP, SpinRAG, SpinType
from tests.fakes import FakeBackend, ledger_spin_map, sample_corpus


def test_epochs_zero_does_not_mint_docs():
    backend = FakeBackend(spin_by_substring=ledger_spin_map())
    rag = SpinRAG(
        content=sample_corpus(),
        n_epochs=0,
        backend=BACKEND_LLAMACPP,
        backend_instance=backend,
    )
    minted = [d for d in rag.documents if d.epoch_history[0]["epoch"] != 0]
    assert minted == []


def test_evolution_creates_top_from_left_and_right():
    backend = FakeBackend(spin_by_substring=ledger_spin_map(), fusion_text="Restored Acme Corp Q1")
    rag = SpinRAG(
        content=sample_corpus(),
        n_epochs=1,
        backend=BACKEND_LLAMACPP,
        backend_instance=backend,
    )
    new_tops = [
        d for d in rag.documents if d.spin == SpinType.TOP and d.epoch_history[0].get("epoch") == 1
    ]
    assert new_tops
    reasons = {d.epoch_history[0].get("reason") for d in new_tops}
    assert "LEFT+TOP transformation" in reasons
    assert "TOP+RIGHT resonance" in reasons
    assert rag.graph["edges"]


def test_evolution_halts_when_fusion_empty():
    backend = FakeBackend(spin_by_substring=ledger_spin_map(), fusion_text="")
    rag = SpinRAG(
        content=sample_corpus(),
        n_epochs=5,
        backend=BACKEND_LLAMACPP,
        backend_instance=backend,
    )
    assert any("Halting evolution" in line for line in rag.get_verbose_log())


def test_make_doc_skips_blank():
    backend = FakeBackend(default_spin="TOP")
    rag = SpinRAG(
        content="only top",
        n_epochs=0,
        backend=BACKEND_LLAMACPP,
        backend_instance=backend,
        auto_init=True,
    )
    assert rag._make_doc("", SpinType.TOP, 1, "x", []) is None
    doc = rag._make_doc("hello", SpinType.TOP, 1, "x", [(rag.documents[0].id, "transforms")])
    assert doc is not None
    assert doc.text == "hello"
    assert rag.graph["edges"][-1]["label"] == "transforms"


def test_closest_doc_empty_base():
    backend = FakeBackend()
    rag = SpinRAG(
        content="only",
        n_epochs=0,
        backend=BACKEND_LLAMACPP,
        backend_instance=backend,
    )
    assert rag._find_closest_doc(rag.documents[0], []) is None


def test_evolution_bottom_rules():
    backend = FakeBackend(
        spin_by_substring={
            "long memo": "BOTTOM",
            "Acme Corp. -": "LEFT",
            "Vendor: Initech": "RIGHT",
        },
        default_spin="TOP",
        fusion_text="Crystallized complex entry",
    )
    rag = SpinRAG(
        content=(
            "long memo about several open ledger items that need structure\n"
            "Acme Corp. -\n"
            "Vendor: Initech | Period: 2026 | Type: revenue\n"
        ),
        n_epochs=1,
        backend=BACKEND_LLAMACPP,
        backend_instance=backend,
    )
    reasons = {
        d.epoch_history[0].get("reason")
        for d in rag.documents
        if d.epoch_history[0].get("epoch") == 1
    }
    assert "BOTTOM+RIGHT combination" in reasons
    assert "BOTTOM+LEFT combination" in reasons


def test_llamacpp_rewrites_default_model_names():
    backend = FakeBackend()
    rag = SpinRAG(
        content="row",
        n_epochs=0,
        backend=BACKEND_LLAMACPP,
        backend_instance=backend,
    )
    assert rag.llm_model == "llama"
    assert rag.embed_model == "llama"
