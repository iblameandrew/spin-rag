"""SpinRAG core: an evolving knowledge-graph RAG focused on document restoration.

The algorithm models each line of source text as a particle with a ``SpinType``.
Through epochs of catalysed interactions, incomplete (``LEFT``) and definitional
(``RIGHT``) fragments are fused with primary (``TOP``) and complex (``BOTTOM``)
documents, gradually reconstructing the latent self-contained definitions
implied by the corpus.

This alpha release is intentionally biased toward *document restoration* over
synthesis: the LLM is prompted to preserve every fact it sees, and pure-TOP
queries are returned verbatim with no LLM rewriting. This trades some
hallucination risk on damaged inputs for high fidelity on intact ones.
"""

from __future__ import annotations

import re
import uuid
import threading
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from langchain_ollama import OllamaLLM as _OllamaLLM
    from langchain_ollama import OllamaEmbeddings
except ImportError:  # pragma: no cover - fallback for older installs
    from langchain_community.llms import Ollama as _OllamaLLM  # type: ignore
    from langchain_community.embeddings import OllamaEmbeddings  # type: ignore


_DEFAULT_EMBED_MODEL = "nomic-embed-text"

_SPIN_PATTERN = re.compile(r"\b(TOP|BOTTOM|LEFT|RIGHT)\b", re.IGNORECASE)


class SpinType(Enum):
    TOP = "TOP"
    BOTTOM = "BOTTOM"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@dataclass
class Document:
    id: str
    text: str
    spin: SpinType
    embeddings: Optional[np.ndarray] = None
    epoch_history: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity that is safe against zero-norm vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _clean_llm_output(text: str) -> str:
    """Trim whitespace, surrounding quotes and stray markdown fences."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
        cleaned = cleaned[1:-1].strip()
    return cleaned


class SpinRAG:
    """Evolving knowledge-graph RAG biased toward document restoration."""

    # Restoration-first prompts: the LLM is repeatedly told to preserve every
    # fact it sees and to mark uncertainty rather than invent.
    _RESTORATION_GUARDRAIL = (
        "You are restoring a damaged knowledge base. Preserve EVERY noun, "
        "proper name, number and relationship from the source fragments "
        "verbatim. Do not invent facts that are not implied by the sources. "
        "If information is genuinely missing, leave a short explicit gap "
        "marker like '[unknown]' instead of guessing. Reply with the "
        "restored statement only, no preamble."
    )

    def __init__(
        self,
        content: str,
        n_epochs: int = 5,
        llm_model: str = "llama2",
        embed_model: Optional[str] = None,
        config: Optional[Dict] = None,
        logger_callback: Optional[Callable[[str], None]] = None,
    ):
        self.content = content
        self.n_epochs = int(n_epochs) if n_epochs is not None else 0
        self.config = config or {}
        self.llm = _OllamaLLM(model=llm_model)
        # Most chat-tuned models (qwen3-*-instruct, llama-*-chat, etc.) do not
        # expose an embeddings endpoint and will crash initialization. Default
        # to a dedicated embedding model unless the caller overrides it.
        self.embed_model_name = embed_model or _DEFAULT_EMBED_MODEL
        self.embeddings_model = OllamaEmbeddings(model=self.embed_model_name)
        self.logger_callback = logger_callback

        # Core state
        self.documents: List[Document] = []
        self.doc_map: Dict[str, Document] = {}
        self.graph: Dict[str, Any] = {"nodes": {}, "edges": []}
        self.verbose_log: List[str] = []

        # Caches and locks
        self._embed_cache: Dict[str, np.ndarray] = {}
        self._embed_lock = threading.Lock()

        self.initialize_index()

    # ------------------------------------------------------------------ utils

    def _log(self, message: str):
        if self.logger_callback:
            try:
                self.logger_callback(message)
            except Exception:  # pragma: no cover - logger must never crash RAG
                pass
        else:
            try:
                print(message)
            except UnicodeEncodeError:
                # Windows consoles default to cp1252 which cannot encode the
                # emoji used in our log lines. Fall back to ascii-safe output.
                try:
                    print(message.encode("ascii", "replace").decode("ascii"))
                except Exception:
                    pass
        self.verbose_log.append(message)

    def _embed(self, text: str) -> np.ndarray:
        """Embed a single string with memoization."""
        key = text or ""
        with self._embed_lock:
            cached = self._embed_cache.get(key)
            if cached is not None:
                return cached
        try:
            vec = np.asarray(self.embeddings_model.embed_query(key), dtype=np.float64)
        except Exception as exc:  # pragma: no cover - depends on Ollama runtime
            self._log(
                f"⚠️ Embedding failed for text of length {len(key)}: {exc}. "
                "Falling back to zero vector."
            )
            vec = np.zeros(1, dtype=np.float64)
        with self._embed_lock:
            self._embed_cache[key] = vec
        return vec

    def _embed_doc(self, doc: Document) -> np.ndarray:
        if doc.embeddings is None or len(doc.embeddings) == 0:
            doc.embeddings = self._embed(doc.text)
        return doc.embeddings

    # ----------------------------------------------------------------- spins

    def _get_spin(self, text: str) -> SpinType:
        prompt = (
            "You are a semantic parser. Classify the text into exactly ONE "
            "of these four labels:\n"
            "- TOP: the text is a self-contained name or concept.\n"
            "- BOTTOM: the text is complex / a target for further evolution.\n"
            "- LEFT: the text is incomplete and is missing information.\n"
            "- RIGHT: the text is a definition / parameter structure.\n\n"
            f'Text: "{text}"\n\n'
            "Answer with a single word (TOP, BOTTOM, LEFT, or RIGHT)."
        )
        try:
            raw = self.llm.invoke(prompt)
        except Exception as exc:
            self._log(f"⚠️ Spin classification failed ({exc}); defaulting to TOP.")
            return SpinType.TOP
        response = _clean_llm_output(raw).upper()
        match = _SPIN_PATTERN.search(response)
        if match:
            return SpinType[match.group(1).upper()]
        return SpinType.TOP

    # ------------------------------------------------------------ initialize

    def initialize_index(self):
        self._log("🚀 Initializing SpinRAG index from in-memory content.")

        raw_text = self.content or ""

        # Line-delimited input: every non-empty line becomes a document so that
        # incomplete fragments stay separated from complete definitions.
        texts = [line.strip() for line in raw_text.split("\n") if line.strip()]
        self._log(f"Found {len(texts)} individual lines to process as documents.")

        if not texts:
            self._log("⚠️ No content to index; skipping evolution.")
            return

        for text in texts:
            doc_id = f"doc_{uuid.uuid4()}"
            spin = self._get_spin(text)
            self._log(f"📄 Chunk '{text}': Assigning initial spin -> {spin.value}")

            doc = Document(
                id=doc_id,
                text=text,
                spin=spin,
                epoch_history=[{"epoch": 0, "spin": spin.value}],
            )
            self.documents.append(doc)
            self.doc_map[doc_id] = doc
            self.graph["nodes"][doc_id] = {
                "text": text,
                "spin": spin.value,
                "epoch_history": doc.epoch_history,
            }

        self.evolve_epochs()
        self._generate_top_spin_embeddings()
        self._log("✅ Index initialization complete.")

    # ----------------------------------------------------------- find closest

    def _find_closest_doc(
        self,
        catalyst_doc: Document,
        base_docs: List[Document],
    ) -> Optional[Document]:
        if not base_docs:
            return None

        catalyst_embedding = self._embed(catalyst_doc.text)
        similarities = [
            _cosine_similarity(catalyst_embedding, self._embed(doc.text))
            for doc in base_docs
        ]

        if not similarities:
            return None

        closest_index = int(np.argmax(similarities))
        return base_docs[closest_index]

    # ----------------------------------------------------------------- evolve

    def _make_doc(
        self,
        text: str,
        spin: SpinType,
        epoch: int,
        reason: str,
        sources: List[Tuple[str, str]],
    ) -> Optional[Document]:
        cleaned = _clean_llm_output(text)
        if not cleaned:
            return None
        new_doc_id = f"doc_{uuid.uuid4()}"
        new_doc = Document(
            id=new_doc_id,
            text=cleaned,
            spin=spin,
            epoch_history=[{"epoch": epoch, "spin": spin.value, "reason": reason}],
        )
        for src_id, label in sources:
            self.graph["edges"].append(
                {"source": src_id, "target": new_doc_id, "label": label}
            )
        return new_doc

    def evolve_epochs(self):
        self._log(f"🔄 Starting evolution for {self.n_epochs} epochs...")

        if self.n_epochs <= 0:
            self._log("  - n_epochs<=0, skipping evolution.")
            return

        # LEFT/RIGHT catalysts are determined once from the source corpus and
        # remain stable; TOP/BOTTOM bases grow as the graph evolves.
        left_catalysts = [d for d in self.documents if d.spin == SpinType.LEFT]
        right_catalysts = [d for d in self.documents if d.spin == SpinType.RIGHT]
        current_top_docs = [d for d in self.documents if d.spin == SpinType.TOP]
        current_bottom_docs = [d for d in self.documents if d.spin == SpinType.BOTTOM]

        for epoch in range(1, self.n_epochs + 1):
            self._log(f"\nEpoch {epoch}:")

            next_gen_top_docs: List[Document] = []
            next_gen_bottom_docs: List[Document] = []

            # Rule: LEFT (catalyst) + TOP (base) -> new TOP
            for left_doc in left_catalysts:
                closest_top_doc = self._find_closest_doc(left_doc, current_top_docs)
                if not closest_top_doc:
                    continue
                self._log(
                    f"  - Transformation: {left_doc.id} (LEFT) with closest "
                    f"{closest_top_doc.id} (TOP) -> TOP"
                )
                prompt = (
                    f"{self._RESTORATION_GUARDRAIL}\n\n"
                    f"Self-contained source: '{closest_top_doc.text}'\n"
                    f"Incomplete fragment:    '{left_doc.text}'\n\n"
                    "Reconstruct a single self-contained statement that keeps "
                    "every fact from BOTH sources. Do not introduce new "
                    "entities."
                )
                new_text = self.llm.invoke(prompt)
                new_doc = self._make_doc(
                    new_text,
                    SpinType.TOP,
                    epoch,
                    "LEFT+TOP transformation",
                    [
                        (left_doc.id, "transforms"),
                        (closest_top_doc.id, "transforms"),
                    ],
                )
                if new_doc:
                    next_gen_top_docs.append(new_doc)
                    self._log(f"    - Created new document {new_doc.text} (TOP)")

            # Rule: RIGHT (catalyst) + TOP (base) -> new TOP
            for right_doc in right_catalysts:
                closest_top_doc = self._find_closest_doc(right_doc, current_top_docs)
                if not closest_top_doc:
                    continue
                self._log(
                    f"  - Resonance: {closest_top_doc.id} (TOP) with "
                    f"{right_doc.id} (RIGHT) -> TOP"
                )
                prompt = (
                    f"{self._RESTORATION_GUARDRAIL}\n\n"
                    f"Self-contained source: '{closest_top_doc.text}'\n"
                    f"Definition/parameters: '{right_doc.text}'\n\n"
                    "Restore a single self-contained definition that merges "
                    "both sources without losing detail from either."
                )
                new_text = self.llm.invoke(prompt)
                new_doc = self._make_doc(
                    new_text,
                    SpinType.TOP,
                    epoch,
                    "TOP+RIGHT resonance",
                    [
                        (right_doc.id, "resonance"),
                        (closest_top_doc.id, "resonance"),
                    ],
                )
                if new_doc:
                    next_gen_top_docs.append(new_doc)
                    self._log(f"    - Spanned new document {new_doc.text} (TOP)")

            # Rule: RIGHT (catalyst) + BOTTOM (base) -> new TOP
            for right_doc in right_catalysts:
                closest_bottom_doc = self._find_closest_doc(
                    right_doc, current_bottom_docs
                )
                if not closest_bottom_doc:
                    continue
                self._log(
                    f"  - Combination: {closest_bottom_doc.id} (BOTTOM) with "
                    f"{right_doc.id} (RIGHT) -> TOP"
                )
                prompt = (
                    f"{self._RESTORATION_GUARDRAIL}\n\n"
                    f"Complex source:        '{closest_bottom_doc.text}'\n"
                    f"Definition/parameters: '{right_doc.text}'\n\n"
                    "Distil a single self-contained concept that preserves "
                    "the specifics of both sources."
                )
                new_text = self.llm.invoke(prompt)
                new_doc = self._make_doc(
                    new_text,
                    SpinType.TOP,
                    epoch,
                    "BOTTOM+RIGHT combination",
                    [
                        (right_doc.id, "combination"),
                        (closest_bottom_doc.id, "combination"),
                    ],
                )
                if new_doc:
                    next_gen_top_docs.append(new_doc)
                    self._log(f"    - Created new document {new_doc.text} (TOP)")

            # Rule: LEFT (catalyst) + BOTTOM (base) -> new BOTTOM
            for left_doc in left_catalysts:
                closest_bottom_doc = self._find_closest_doc(
                    left_doc, current_bottom_docs
                )
                if not closest_bottom_doc:
                    continue
                self._log(
                    f"  - Combination: {closest_bottom_doc.id} (BOTTOM) with "
                    f"{left_doc.id} (LEFT) -> BOTTOM"
                )
                prompt = (
                    f"{self._RESTORATION_GUARDRAIL}\n\n"
                    f"Complex source:     '{closest_bottom_doc.text}'\n"
                    f"Incomplete fragment:'{left_doc.text}'\n\n"
                    "Produce a richer complex target that still preserves "
                    "every specific fact present in both sources."
                )
                new_text = self.llm.invoke(prompt)
                new_doc = self._make_doc(
                    new_text,
                    SpinType.BOTTOM,
                    epoch,
                    "BOTTOM+LEFT combination",
                    [
                        (left_doc.id, "combination"),
                        (closest_bottom_doc.id, "combination"),
                    ],
                )
                if new_doc:
                    next_gen_bottom_docs.append(new_doc)
                    self._log(f"    - Created new document {new_doc.text} (BOTTOM)")

            if not next_gen_top_docs and not next_gen_bottom_docs:
                self._log(
                    "  - No new documents produced in this epoch. Halting evolution."
                )
                break

            new_docs_this_epoch = next_gen_top_docs + next_gen_bottom_docs
            self.documents.extend(new_docs_this_epoch)
            for doc in new_docs_this_epoch:
                self.doc_map[doc.id] = doc
                self.graph["nodes"][doc.id] = {
                    "text": doc.text,
                    "spin": doc.spin.value,
                    "epoch_history": doc.epoch_history,
                }

            current_top_docs.extend(next_gen_top_docs)
            current_bottom_docs.extend(next_gen_bottom_docs)

        self._log("\n✅ Evolution complete.")

    # --------------------------------------------------------- embeddings

    def _generate_top_spin_embeddings(self):
        self._log("🧠 Generating embeddings for TOP spin documents...")
        for doc in self.documents:
            if doc.spin == SpinType.TOP:
                self._embed_doc(doc)
        self._log("✅ Embeddings generated.")

    # --------------------------------------------------------------- query

    def query(self, query_text: str, reorganize_graph: bool = False) -> str:
        self._log(f"\n🔍 Query received: '{query_text}'")
        if not query_text or not query_text.strip():
            return ""

        query_spin = self._get_spin(query_text)
        self._log(f"  - Query spin detected as: {query_spin.value}")
        query_embedding = self._embed(query_text)

        top_spin_docs = [
            d
            for d in self.documents
            if d.spin == SpinType.TOP
            and d.embeddings is not None
            and len(d.embeddings) > 0
        ]
        if not top_spin_docs:
            return "No TOP spin documents available for querying."

        similarities = [
            (doc, _cosine_similarity(query_embedding, doc.embeddings))
            for doc in top_spin_docs
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)

        closest_doc, similarity = similarities[0]
        self._log(
            f"  - Closest TOP document: {closest_doc.id} (Similarity: {similarity:.4f})"
        )

        query_doc_id = f"doc_{uuid.uuid4()}"
        if reorganize_graph:
            self._log("  - Reorganizing graph with new data from query.")
            new_doc = Document(
                id=query_doc_id,
                text=query_text,
                spin=query_spin,
                epoch_history=[
                    {
                        "epoch": "query",
                        "spin": query_spin.value,
                        "reason": "Query input",
                    }
                ],
            )
            self.documents.append(new_doc)
            self.doc_map[new_doc.id] = new_doc
            self.graph["nodes"][new_doc.id] = {
                "text": new_doc.text,
                "spin": new_doc.spin.value,
            }
            self.graph["edges"].append(
                {"source": query_doc_id, "target": closest_doc.id, "label": "queries"}
            )

        response_text = self._apply_query_production_rules(
            query_text, query_spin, query_embedding, closest_doc
        )
        response_text = _clean_llm_output(response_text) or closest_doc.text

        if reorganize_graph:
            response_doc_id = f"doc_{uuid.uuid4()}"
            response_spin = self._get_spin(response_text)
            response_doc = Document(
                id=response_doc_id,
                text=response_text,
                spin=response_spin,
                epoch_history=[
                    {
                        "epoch": "query_response",
                        "spin": response_spin.value,
                        "reason": "Generated from query",
                    }
                ],
            )
            self.documents.append(response_doc)
            self.doc_map[response_doc.id] = response_doc
            self.graph["nodes"][response_doc.id] = {
                "text": response_doc.text,
                "spin": response_doc.spin.value,
            }
            self.graph["edges"].append(
                {
                    "source": query_doc_id,
                    "target": response_doc_id,
                    "label": "generates_response",
                }
            )
            self.graph["edges"].append(
                {
                    "source": closest_doc.id,
                    "target": response_doc_id,
                    "label": "influences_response",
                }
            )

        return response_text

    def _apply_query_production_rules(
        self,
        query_text: str,
        query_spin: SpinType,
        query_embedding: np.ndarray,
        top_doc: Document,
    ) -> str:
        if query_spin == SpinType.LEFT:
            self._log("  - Applying rule: LEFT (query) + TOP (doc) -> RIGHT")
            prompt = (
                f"{self._RESTORATION_GUARDRAIL}\n\n"
                f"Self-contained source: '{top_doc.text}'\n"
                f"Incomplete query:      '{query_text}'\n\n"
                "Restore the missing parameter structure implied by the source."
            )
            return self.llm.invoke(prompt)

        if query_spin == SpinType.RIGHT:
            self._log("  - Applying rule: RIGHT (query) + TOP (doc) -> BOTTOM")
            prompt = (
                f"{self._RESTORATION_GUARDRAIL}\n\n"
                f"Self-contained source: '{top_doc.text}'\n"
                f"Definition/parameters: '{query_text}'\n\n"
                "Project a single complex target that preserves every fact in "
                "both sources."
            )
            return self.llm.invoke(prompt)

        if query_spin == SpinType.BOTTOM:
            adjacent_docs = self._get_adjacent_docs(top_doc.id)
            self._log(
                f"  - Found {len(adjacent_docs)} documents adjacent to the "
                f"closest TOP doc ({top_doc.id})."
            )

            closest_left_doc, left_sim = self._find_closest_doc_of_spin(
                query_embedding, SpinType.LEFT, search_space=adjacent_docs
            )
            closest_right_doc, right_sim = self._find_closest_doc_of_spin(
                query_embedding, SpinType.RIGHT, search_space=adjacent_docs
            )

            if closest_right_doc and right_sim >= left_sim and right_sim > 0:
                self._log(
                    f"  - Applying rule: BOTTOM (query) + RIGHT "
                    f"(doc: {closest_right_doc.id}, sim: {right_sim:.4f}) -> TOP"
                )
                prompt = (
                    f"{self._RESTORATION_GUARDRAIL}\n\n"
                    f"Evolutionary target:   '{query_text}'\n"
                    f"Definition/parameters: '{closest_right_doc.text}'\n\n"
                    "Restore a single self-contained concept that keeps every "
                    "specific from both sources."
                )
                return self.llm.invoke(prompt)

            if closest_left_doc and left_sim > 0:
                self._log(
                    f"  - Applying rule: BOTTOM (query) + LEFT "
                    f"(doc: {closest_left_doc.id}, sim: {left_sim:.4f}) -> RIGHT"
                )
                prompt = (
                    f"{self._RESTORATION_GUARDRAIL}\n\n"
                    f"Evolutionary target: '{query_text}'\n"
                    f"Incomplete fragment: '{closest_left_doc.text}'\n\n"
                    "Restore the missing parameter structure without inventing "
                    "new entities."
                )
                return self.llm.invoke(prompt)

        # Restoration default: return the closest TOP doc verbatim.
        return top_doc.text

    # --------------------------------------------------- search-space helpers

    def _find_closest_doc_of_spin(
        self,
        query_embedding: np.ndarray,
        spin_type: SpinType,
        search_space: Optional[List[Document]] = None,
    ) -> Tuple[Optional[Document], float]:
        source_docs = self.documents if search_space is None else search_space
        target_docs = [d for d in source_docs if d.spin == spin_type]
        if not target_docs:
            return None, 0.0

        similarities = [
            _cosine_similarity(query_embedding, self._embed(doc.text))
            for doc in target_docs
        ]
        if not similarities:
            return None, 0.0

        max_index = int(np.argmax(similarities))
        return target_docs[max_index], float(similarities[max_index])

    def _get_adjacent_docs(self, doc_id: str) -> List[Document]:
        adjacent_ids = set()
        for edge in self.graph["edges"]:
            if edge.get("source") == doc_id:
                adjacent_ids.add(edge.get("target"))
            elif edge.get("target") == doc_id:
                adjacent_ids.add(edge.get("source"))

        return [
            self.doc_map[adj_id] for adj_id in adjacent_ids if adj_id in self.doc_map
        ]

    # --------------------------------------------------------------- logging

    def get_verbose_log(self) -> List[str]:
        # Return a snapshot so callers cannot mutate internal state.
        return list(self.verbose_log)

    def clear_log(self):
        self.verbose_log = []


__all__ = ["SpinRAG", "SpinType", "Document"]
