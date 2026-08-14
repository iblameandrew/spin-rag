from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional


class FakeBackend:
    """Deterministic stand-in for `_Backend` — no sockets."""

    def __init__(
        self,
        spin_by_substring: Optional[Dict[str, str]] = None,
        default_spin: str = "TOP",
        fusion_text: str = "Restored: merged fragment with complete entry",
        chat_error: Optional[Exception] = None,
        embed_error: Optional[Exception] = None,
        dim: int = 8,
    ) -> None:
        self.base_url = "http://fake.local/v1"
        self.spin_by_substring = spin_by_substring or {}
        self.default_spin = default_spin
        self.fusion_text = fusion_text
        self.chat_error = chat_error
        self.embed_error = embed_error
        self.dim = dim
        self.chat_prompts: List[str] = []
        self.embed_calls: List[List[str]] = []

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if self.chat_error:
            raise self.chat_error
        prompt = messages[-1]["content"] if messages else ""
        self.chat_prompts.append(prompt)
        if "Classify the text" in prompt:
            return self._classify(prompt)
        return self.fusion_text

    def embed(self, model: str, inputs: Any) -> List[List[float]]:
        if self.embed_error:
            raise self.embed_error
        if isinstance(inputs, str):
            inputs = [inputs]
        texts = [t for t in inputs if t]
        self.embed_calls.append(list(texts))
        return [self._vec(text) for text in texts]

    def _classify(self, prompt: str) -> str:
        for needle, spin in self.spin_by_substring.items():
            if needle in prompt:
                return spin
        return self.default_spin

    def _vec(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(digest[i] / 255.0) + i * 0.001 for i in range(self.dim)]


def ledger_spin_map() -> Dict[str, str]:
    """Heuristic labels matching demo-data.txt roles."""
    return {
        "Acme Corp. -": "LEFT",
        "Consulting": "LEFT",
        "Pied Piper": "LEFT",
        "Datadog - observability": "LEFT",
        "Vendor: Initech": "RIGHT",
        "Summarize": "BOTTOM",
        "What revenue": "BOTTOM",
    }


def sample_corpus() -> str:
    return (
        "Revenue 2026-Q1-006 | Client: Globex Industries | Period: 2026-Q1 | Amount: $128,000.00\n"
        "Revenue 2026-Q1-003 | Client: Acme Corp. | Period: 2026-Q1 | Amount: $45,200.00 | Product: Consulting\n"
        "Acme Corp. -\n"
        "Consulting\n"
        "Vendor: Initech | Period: 2026 | Type: revenue\n"
        "Pied Piper\n"
    )
