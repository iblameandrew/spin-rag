from __future__ import annotations

import pytest

from tests.fakes import FakeBackend, ledger_spin_map, sample_corpus


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend(spin_by_substring=ledger_spin_map())


@pytest.fixture
def corpus() -> str:
    return sample_corpus()
