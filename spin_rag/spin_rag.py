# spinlm.py
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import uuid
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# ========================
# Core Components & Enums
# ========================

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

# ========================
# Main SpinRAG Class
# ========================

class SpinRAG:
    def __init__(self, file_path: str, n_epochs: int = 5, llm_model: str = "llama2", config: Optional[Dict] = None):
        self.file_path = Path(file_path)
        self.n_epochs = n_epochs
        self.config = config or {}
        self.llm = Ollama(model=llm_model)
        self.embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Core state
        self.documents: List[Document] = []
        self.graph: Dict[str, Any] = {"nodes": {}, "edges": []}
        self.verbose_log: List[str] = []

        self.initialize_index()

    def _log(self, message: str):
        print(message)
        self.verbose_log.append(message)

    def _get_spin(self, text: str) -> SpinType:
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Analyze the following text and determine its spin type based on these definitions:
            - TOP: The text is self-contained and does not refer to external concepts.
            - BOTTOM: The text represents a potential evolution or target for other data.
            - LEFT: The text contains partial definitions or is incomplete.
            - RIGHT: The text describes a parameter structure or a component that can be combined with other data.

            Text: "{text}"

            Based on the definitions, the spin type is:
            """
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(text=text)
        
        if "TOP" in response:
            return SpinType.TOP
        elif "BOTTOM" in response:
            return SpinType.BOTTOM
        elif "LEFT" in response:
            return SpinType.LEFT
        elif "RIGHT" in response:
            return SpinType.RIGHT
        return SpinType.TOP

    def initialize_index(self):
        self._log(f"🚀 Initializing SpinRAG index from: {self.file_path}")
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with open(self.file_path, 'r') as f:
            raw_text = f.read()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.split_text(raw_text)

        for i, text in enumerate(texts):
            doc_id = f"doc_{uuid.uuid4()}"
            spin = self._get_spin(text)
            self._log(f"📄 Chunk {i}: Assigning initial spin -> {spin.value}")
            
            doc = Document(
                id=doc_id,
                text=text,
                spin=spin,
                epoch_history=[{"epoch": 0, "spin": spin.value}]
            )
            self.documents.append(doc)
            self.graph["nodes"][doc_id] = {
                "text": text,
                "spin": spin.value,
                "epoch_history": doc.epoch_history
            }
        
        self.evolve_epochs()
        self._generate_top_spin_embeddings()
        self._log("✅ Index initialization complete.")

    def evolve_epochs(self):
        self._log(f"🔄 Starting evolution for {self.n_epochs} epochs...")
        for epoch in range(1, self.n_epochs + 1):
            self._log(f"\nEpoch {epoch}:")
            new_documents = []
            for i in range(len(self.documents)):
                for j in range(len(self.documents)):
                    if i == j:
                        continue

                    doc1 = self.documents[i]
                    doc2 = self.documents[j]

                    # TOP repels BOTTOM, LEFT repels RIGHT
                    if (doc1.spin == SpinType.TOP and doc2.spin == SpinType.BOTTOM) or \
                       (doc1.spin == SpinType.LEFT and doc2.spin == SpinType.RIGHT):
                        self._log(f"  - Repulsion between {doc1.id} ({doc1.spin.value}) and {doc2.id} ({doc2.spin.value})")

                    # TOP has resonance with RIGHT
                    if doc1.spin == SpinType.TOP and doc2.spin == SpinType.RIGHT:
                        self._log(f"  - Resonance: {doc1.id} (TOP) with {doc2.id} (RIGHT)")
                        # Generates a projection where RIGHT combines with TOP and spans a BOTTOM
                        new_text_prompt = f"Combine the self-contained idea '{doc1.text}' with the parameter structure '{doc2.text}' to create a new potential evolution or target."
                        new_text = self.llm.predict(new_text_prompt)
                        new_doc_id = f"doc_{uuid.uuid4()}"
                        new_doc = Document(id=new_doc_id, text=new_text, spin=SpinType.BOTTOM, epoch_history=[{"epoch": epoch, "spin": SpinType.BOTTOM.value, "reason": "TOP+RIGHT resonance"}])
                        new_documents.append(new_doc)
                        self._log(f"    - Spanned new document {new_doc_id} (BOTTOM)")

                    # LEFT with TOP turns LEFT into RIGHT
                    if doc1.spin == SpinType.LEFT and doc2.spin == SpinType.TOP:
                        self._log(f"  - Transformation: {doc1.id} (LEFT) with {doc2.id} (TOP) -> RIGHT")
                        doc1.spin = SpinType.RIGHT
                        doc1.epoch_history.append({"epoch": epoch, "spin": SpinType.RIGHT.value, "reason": "LEFT+TOP interaction"})

                    # BOTTOM combines with RIGHT to create a TOP
                    if doc1.spin == SpinType.BOTTOM and doc2.spin == SpinType.RIGHT:
                        self._log(f"  - Combination: {doc1.id} (BOTTOM) with {doc2.id} (RIGHT) -> TOP")
                        new_text_prompt = f"Combine the evolutionary target '{doc1.text}' with the parameter structure '{doc2.text}' to create a new self-contained concept."
                        new_text = self.llm.predict(new_text_prompt)
                        new_doc_id = f"doc_{uuid.uuid4()}"
                        new_doc = Document(id=new_doc_id, text=new_text, spin=SpinType.TOP, epoch_history=[{"epoch": epoch, "spin": SpinType.TOP.value, "reason": "BOTTOM+RIGHT combination"}])
                        new_documents.append(new_doc)
                        self._log(f"    - Created new document {new_doc_id} (TOP)")
            
            self.documents.extend(new_documents)
            for doc in self.documents:
                self.graph["nodes"][doc.id] = {
                    "text": doc.text,
                    "spin": doc.spin.value,
                    "epoch_history": doc.epoch_history
                }
        self._log("\n✅ Evolution complete.")

    def _generate_top_spin_embeddings(self):
        self._log("🧠 Generating embeddings for TOP spin documents...")
        for doc in self.documents:
            if doc.spin == SpinType.TOP:
                doc.embeddings = np.array(self.embeddings_model.embed_query(doc.text))
        self._log("✅ Embeddings generated.")

    def query(self, query_text: str, reorganize_graph: bool = False) -> str:
        self._log(f"\n🔍 Query received: '{query_text}'")
        query_spin = self._get_spin(query_text)
        self._log(f"  - Query spin detected as: {query_spin.value}")
        query_embedding = np.array(self.embeddings_model.embed_query(query_text))

        top_spin_docs = [doc for doc in self.documents if doc.spin == SpinType.TOP and doc.embeddings is not None]
        if not top_spin_docs:
            return "No TOP spin documents available for querying."

        similarities = [(doc, np.dot(query_embedding, doc.embeddings) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc.embeddings))) for doc in top_spin_docs]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        closest_doc, similarity = similarities[0]
        self._log(f"  - Closest TOP document: {closest_doc.id} (Similarity: {similarity:.4f})")
        
        # Data acceleration
        if query_spin == SpinType.BOTTOM:
            self._log("  - ⚠️ High acceleration detected (Query is BOTTOM to closest TOP)!")
            if len(similarities) > 1:
                closest_doc, similarity = similarities[1]
                self._log(f"  - Slingshot to next closest TOP document: {closest_doc.id}")
            else:
                self._log("  - Not enough TOP documents to perform slingshot.")

        # Apply production rules
        response = self._apply_query_production_rules(query_text, query_spin, closest_doc)
        
        if reorganize_graph:
            self._log("  - Reorganizing graph with new data from query.")
            new_doc_id = f"doc_{uuid.uuid4()}"
            new_doc = Document(id=new_doc_id, text=query_text, spin=query_spin, epoch_history=[{"epoch": "query", "spin": query_spin.value, "reason": "Query input"}])
            self.documents.append(new_doc)
            self.graph["nodes"][new_doc_id] = {"text": new_doc.text, "spin": new_doc.spin.value}
            # A full re-evolution could be triggered here
            # self.evolve_epochs() 
            # self._generate_top_spin_embeddings()

        return response

    def _apply_query_production_rules(self, query_text: str, query_spin: SpinType, top_doc: Document) -> str:
        if query_spin == SpinType.LEFT:
            # LEFT with TOP turns LEFT into RIGHT
            self._log("  - Applying rule: LEFT (query) + TOP (doc) -> RIGHT")
            new_text_prompt = f"Based on the self-contained concept '{top_doc.text}', complete the partial definition '{query_text}' into a structured parameter."
            return self.llm.predict(new_text_prompt)
        elif query_spin == SpinType.RIGHT:
            # TOP has resonance with RIGHT and spans a BOTTOM
            self._log("  - Applying rule: RIGHT (query) + TOP (doc) -> BOTTOM")
            new_text_prompt = f"Combine the self-contained idea '{top_doc.text}' with the parameter structure '{query_text}' to create a new potential evolution or target."
            return self.llm.predict(new_text_prompt)
        else:
            # Default to closest match
            return top_doc.text

    def get_verbose_log(self) -> List[str]:
        return self.verbose_log

    def clear_log(self):
        self.verbose_log = []

# ========================
# Export Public API
# ========================

__all__ = [
    "SpinRAG",
    "SpinType",
    "Document"
]