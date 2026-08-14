from pathlib import Path

from spin_rag import BACKEND_LLAMACPP, SpinRAG, SpinType
from tests.fakes import FakeBackend, ledger_spin_map


def test_demo_data_pipeline_end_to_end():
    demo = Path(__file__).resolve().parents[2] / "demo-data.txt"
    content = demo.read_text(encoding="utf-8")
    backend = FakeBackend(spin_by_substring=ledger_spin_map())
    rag = SpinRAG(
        content=content,
        n_epochs=2,
        backend=BACKEND_LLAMACPP,
        backend_instance=backend,
    )
    assert rag.documents
    spins = {d.spin for d in rag.documents}
    assert SpinType.TOP in spins
    assert SpinType.LEFT in spins
    assert SpinType.RIGHT in spins

    verbatim = rag.query("Globex Industries platform license")
    assert verbatim  # TOP path or cleaned fusion
    left = rag.query("Acme Corp. -")
    assert left
    bottom = rag.query("Summarize 2026-Q1 receivables.")
    assert bottom
    log = rag.get_verbose_log()
    assert any("Initializing SpinRAG" in line for line in log)
    assert any("Evolution complete" in line or "Halting" in line for line in log)


def test_auto_init_false_defers_work():
    backend = FakeBackend()
    rag = SpinRAG(
        content="Revenue 1\nAcme -",
        n_epochs=1,
        backend=BACKEND_LLAMACPP,
        backend_instance=backend,
        auto_init=False,
    )
    assert rag.documents == []
    rag.initialize_index()
    assert rag.documents
