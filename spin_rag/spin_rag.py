import hashlib
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import uuid
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

class SpinRAG:
    def __init__(self, content: str, n_epochs: int = 5, llm_model: str = "llama2", config: Optional[Dict] = None, logger_callback: Optional[Callable[[str], None]] = None):
        self.content = content
        self.n_epochs = n_epochs
        self.config = config or {}
        self.llm = Ollama(model=llm_model)
        self.embeddings_model = OllamaEmbeddings(model=llm_model)
        self.logger_callback = logger_callback

        # Core state
        self.documents: List[Document] = []
        self.doc_map: Dict[str, Document] = {}
        self.graph: Dict[str, Any] = {"nodes": {}, "edges": []}
        self.verbose_log: List[str] = []

        self.initialize_index()

    def _log(self, message: str):
        # Use the callback to send the log to the frontend if it exists
        if self.logger_callback:
            self.logger_callback(message)
        else:
            # Fallback to printing if no callback is provided
            print(message)
        self.verbose_log.append(message)

    def _get_spin(self, text: str) -> SpinType:
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            You are a semantic parser. Your task is to classify text by measure of how complex is.

            - TOP: The text is a name.
            - BOTTOM: The text is complex.
            - LEFT: The text is incomplete and is missing some information to be understood.
            - RIGHT: The text is a definition.
            
            Text: "{text}"

            Answer only the type. Based on the definitions, the type is:
            """
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.invoke(input={"text": text})
        response = response["text"]
        
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
        self._log(f"🚀 Initializing SpinRAG index from in-memory content.")
        
        # Use the content string directly instead of reading a file
        raw_text = self.content

        # --- FIX START ---
        # The original RecursiveCharacterTextSplitter was grouping short, incomplete lines.
        # Since the data is line-delimited, we split by newline and filter out empty lines.
        # This ensures every entry, complete or not, is treated as a separate document.
        texts = [line.strip() for line in raw_text.split('\n') if line.strip()]
        self._log(f"Found {len(texts)} individual lines to process as documents.")
        # --- FIX END ---

        for i, text in enumerate(texts):
            doc_id = f"doc_{uuid.uuid4()}"
            spin = self._get_spin(text)
            self._log(f"📄 Chunk '{text}': Assigning initial spin -> {spin.value}")
            
            doc = Document(
                id=doc_id,
                text=text,
                spin=spin,
                epoch_history=[{"epoch": 0, "spin": spin.value}]
            )
            self.documents.append(doc)
            self.doc_map[doc_id] = doc
            self.graph["nodes"][doc_id] = {
                "text": text,
                "spin": spin.value,
                "epoch_history": doc.epoch_history
            }
        
        self.evolve_epochs()
        self._generate_top_spin_embeddings()
        self._log("✅ Index initialization complete.")

    def _find_closest_doc(self, catalyst_doc: Document, base_docs: List[Document]) -> Optional[Document]:
        if not base_docs:
            return None

        catalyst_embedding = self.embeddings_model.embed_query(catalyst_doc.text)
        base_embeddings = self.embeddings_model.embed_documents([doc.text for doc in base_docs])

        similarities = [np.dot(catalyst_embedding, base_emb) / (np.linalg.norm(catalyst_embedding) * np.linalg.norm(base_emb)) for base_emb in base_embeddings]
        
        if not similarities:
            return None

        closest_index = np.argmax(similarities)
        return base_docs[closest_index]

    def evolve_epochs(self):
        self._log(f"🔄 Starting evolution for {self.n_epochs} epochs...")

        # Segregate initial documents. LEFT/RIGHT docs are catalysts, TOP/BOTTOM are the evolving base.
        left_catalysts = [doc for doc in self.documents if doc.spin == SpinType.LEFT]
        right_catalysts = [doc for doc in self.documents if doc.spin == SpinType.RIGHT]
        
        current_top_docs = [doc for doc in self.documents if doc.spin == SpinType.TOP]
        current_bottom_docs = [doc for doc in self.documents if doc.spin == SpinType.BOTTOM]

        for epoch in range(1, self.n_epochs + 1):
            self._log(f"\nEpoch {epoch}:")
            
            # These will hold the documents produced in this epoch.
            next_gen_top_docs = []
            next_gen_bottom_docs = []

            # --- Interaction Phase ---
            # All LEFT/RIGHT catalysts interact with the metrically closest TOP/BOTTOM documents.

            # Rule: LEFT (catalyst) + TOP (base) -> new TOP
            for left_doc in left_catalysts:
                closest_top_doc = self._find_closest_doc(left_doc, current_top_docs)
                if closest_top_doc:
                    self._log(f"  - Transformation: {left_doc.id} (LEFT) with closest {closest_top_doc.id} (TOP) -> TOP")
                    new_text_prompt = f"Combine the self-contained idea '{closest_top_doc.text}' with the incomplete data '{left_doc.text}' to create a new self-contained definition. Answer only with the new concept."
                    new_text = self.llm.invoke(new_text_prompt)
                    new_doc_id = f"doc_{uuid.uuid4()}"
                    new_doc = Document(id=new_doc_id, text=new_text, spin=SpinType.TOP, epoch_history=[{"epoch": epoch, "spin": SpinType.TOP.value, "reason": "LEFT+TOP transformation"}])
                    next_gen_top_docs.append(new_doc)
                    # Add edges to the graph
                    self.graph["edges"].append({"source": left_doc.id, "target": new_doc_id, "label": "transforms"})
                    self.graph["edges"].append({"source": closest_top_doc.id, "target": new_doc_id, "label": "transforms"})
                    self._log(f"    - Created new document {new_text} (TOP)")

            # Rule: RIGHT (catalyst) + TOP (base) -> new TOP
            for right_doc in right_catalysts:
                closest_top_doc = self._find_closest_doc(right_doc, current_top_docs)
                if closest_top_doc:
                    self._log(f"  - Resonance: {closest_top_doc.id} (TOP) with {right_doc.id} (RIGHT) -> TOP")
                    new_text_prompt = f"Combine the self-contained idea '{closest_top_doc.text}' with the parameter structure '{right_doc.text}' to create a new self contained definition. Answer only with the new concept."
                    new_text = self.llm.invoke(new_text_prompt)
                    new_doc_id = f"doc_{uuid.uuid4()}"
                    new_doc = Document(id=new_doc_id, text=new_text, spin=SpinType.TOP, epoch_history=[{"epoch": epoch, "spin": SpinType.TOP.value, "reason": "TOP+RIGHT resonance"}]) 
                    next_gen_top_docs.append(new_doc)
                    # Add edges to the graph
                    self.graph["edges"].append({"source": right_doc.id, "target": new_doc_id, "label": "resonance"})
                    self.graph["edges"].append({"source": closest_top_doc.id, "target": new_doc_id, "label": "resonance"})
                    self._log(f"    - Spanned new document {new_text} (TOP)")

            # Rule: RIGHT (catalyst) + BOTTOM (base) -> new TOP
            for right_doc in right_catalysts:
                closest_bottom_doc = self._find_closest_doc(right_doc, current_bottom_docs)
                if closest_bottom_doc:
                    self._log(f"  - Combination: {closest_bottom_doc.id} (BOTTOM) with {right_doc.id} (RIGHT) -> TOP")
                    new_text_prompt = f"Combine the complex concept '{closest_bottom_doc.text}' with the parameter structure '{right_doc.text}' to create a new short self-contained concept. Answer only with the new concept."
                    new_text = self.llm.invoke(new_text_prompt)
                    new_doc_id = f"doc_{uuid.uuid4()}"
                    new_doc = Document(id=new_doc_id, text=new_text, spin=SpinType.TOP, epoch_history=[{"epoch": epoch, "spin": SpinType.TOP.value, "reason": "BOTTOM+RIGHT combination"}])
                    next_gen_top_docs.append(new_doc)
                    # Add edges to the graph
                    self.graph["edges"].append({"source": right_doc.id, "target": new_doc_id, "label": "combination"})
                    self.graph["edges"].append({"source": closest_bottom_doc.id, "target": new_doc_id, "label": "combination"})
                    self._log(f"    - Created new document {new_text} (TOP)")

            # Rule: LEFT (catalyst) + BOTTOM (base) -> new BOTTOM
            for left_doc in left_catalysts:
                closest_bottom_doc = self._find_closest_doc(left_doc, current_bottom_docs)
                if closest_bottom_doc:
                    self._log(f"  - Combination: {closest_bottom_doc.id} (BOTTOM) with {left_doc.id} (LEFT) -> BOTTOM")
                    new_text_prompt = f"Combine the complex concept '{closest_bottom_doc.text}' with the incomplete '{left_doc.text}' to create a new more complex concept. Answer only with the new concept."
                    new_text = self.llm.invoke(new_text_prompt)
                    new_doc_id = f"doc_{uuid.uuid4()}"
                    new_doc = Document(id=new_doc_id, text=new_text, spin=SpinType.BOTTOM, epoch_history=[{"epoch": epoch, "spin": SpinType.BOTTOM.value, "reason": "BOTTOM+LEFT combination"}])
                    next_gen_bottom_docs.append(new_doc)
                    # Add edges to the graph
                    self.graph["edges"].append({"source": left_doc.id, "target": new_doc_id, "label": "combination"})
                    self.graph["edges"].append({"source": closest_bottom_doc.id, "target": new_doc_id, "label": "combination"})
                    self._log(f"    - Created new document {new_text} (BOTTOM)")
            
            # Check if any evolution occurred.
            if not next_gen_top_docs and not next_gen_bottom_docs:
                self._log("  - No new documents produced in this epoch. Halting evolution.")
                break

            # Add the newly created generation to the global list and the graph.
            new_docs_this_epoch = next_gen_top_docs + next_gen_bottom_docs
            self.documents.extend(new_docs_this_epoch)
            for doc in new_docs_this_epoch:
                self.doc_map[doc.id] = doc
                self.graph["nodes"][doc.id] = {
                    "text": doc.text,
                    "spin": doc.spin.value,
                    "epoch_history": doc.epoch_history
                }
            
            # The next generation becomes the current generation for the next epoch.
            current_top_docs.extend(next_gen_top_docs)
            current_bottom_docs.extend(next_gen_bottom_docs)


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
        
        query_doc_id = f"doc_{uuid.uuid4()}"
        if reorganize_graph:
            self._log("  - Reorganizing graph with new data from query.")
            new_doc = Document(id=query_doc_id, text=query_text, spin=query_spin, epoch_history=[{"epoch": "query", "spin": query_spin.value, "reason": "Query input"}])
            self.documents.append(new_doc)
            self.doc_map[new_doc.id] = new_doc
            self.graph["nodes"][new_doc.id] = {"text": new_doc.text, "spin": new_doc.spin.value}
            self.graph["edges"].append({"source": query_doc_id, "target": closest_doc.id, "label": "queries"})


        # Apply production rules
        response_text = self._apply_query_production_rules(query_text, query_spin, query_embedding, closest_doc)
        
        if reorganize_graph:
            # Add the response as a new node and link it
            response_doc_id = f"doc_{uuid.uuid4()}"
            response_spin = self._get_spin(response_text)
            response_doc = Document(id=response_doc_id, text=response_text, spin=response_spin, epoch_history=[{"epoch": "query_response", "spin": response_spin.value, "reason": "Generated from query"}])
            self.documents.append(response_doc)
            self.doc_map[response_doc.id] = response_doc
            self.graph["nodes"][response_doc.id] = {"text": response_doc.text, "spin": response_doc.spin.value}
            self.graph["edges"].append({"source": query_doc_id, "target": response_doc_id, "label": "generates_response"})
            self.graph["edges"].append({"source": closest_doc.id, "target": response_doc_id, "label": "influences_response"})

        return response_text

    def _apply_query_production_rules(self, query_text: str, query_spin: SpinType, query_embedding: np.ndarray, top_doc: Document) -> str:
        if query_spin == SpinType.LEFT:
            # LEFT with TOP turns LEFT into RIGHT
            self._log("  - Applying rule: LEFT (query) + TOP (doc) -> RIGHT")
            new_text_prompt = f"Based on the self-contained concept '{top_doc.text}', complete the partial definition '{query_text}' into a structured parameter."
            return self.llm.invoke(new_text_prompt)
        
        elif query_spin == SpinType.RIGHT:
            # TOP has resonance with RIGHT and spans a BOTTOM
            self._log("  - Applying rule: RIGHT (query) + TOP (doc) -> BOTTOM")
            new_text_prompt = f"Combine the self-contained idea '{top_doc.text}' with the parameter structure '{query_text}' to create a new potential evolution or target."
            return self.llm.invoke(new_text_prompt)

        elif query_spin == SpinType.BOTTOM:
            # Get the adjacent documents to the closest TOP document
            adjacent_docs = self._get_adjacent_docs(top_doc.id)
            self._log(f"  - Found {len(adjacent_docs)} documents adjacent to the closest TOP doc ({top_doc.id}).")

            # For a BOTTOM query, find the closest LEFT and RIGHT docs *within the adjacent nodes*
            closest_left_doc, left_sim = self._find_closest_doc_of_spin(query_embedding, SpinType.LEFT, search_space=adjacent_docs)
            closest_right_doc, right_sim = self._find_closest_doc_of_spin(query_embedding, SpinType.RIGHT, search_space=adjacent_docs)


            if right_sim > left_sim and closest_right_doc:
                # BOTTOM combines with RIGHT to create a TOP
                self._log(f"  - Applying rule: BOTTOM (query) + RIGHT (doc: {closest_right_doc.id}, sim: {right_sim:.4f}) -> TOP")
                prompt = f"Combine the evolutionary target '{query_text}' with the parameter structure '{closest_right_doc.text}' to create a new self-contained concept."
                return self.llm.invoke(prompt)
            
            elif left_sim > 0 and closest_left_doc:
                # BOTTOM attracts LEFT to create a RIGHT
                self._log(f"  - Applying rule: BOTTOM (query) + LEFT (doc: {closest_left_doc.id}, sim: {left_sim:.4f}) -> RIGHT")
                prompt = f"Using the evolutionary target '{query_text}' as a goal, complete the partial definition '{closest_left_doc.text}' to describe a new parameter structure."
                return self.llm.invoke(prompt)

        # Default to closest match if no other rules apply
        return top_doc.text

    def _find_closest_doc_of_spin(self, query_embedding: np.ndarray, spin_type: SpinType, search_space: Optional[List[Document]] = None) -> Tuple[Optional[Document], float]:
        """Finds the most similar document of a given spin type within a given search space."""
        source_docs = self.documents if search_space is None else search_space
        target_docs = [doc for doc in source_docs if doc.spin == spin_type]
        if not target_docs:
            return None, 0.0

        doc_embeddings = self.embeddings_model.embed_documents([doc.text for doc in target_docs])
        
        similarities = [np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)) for doc_emb in doc_embeddings]
        
        if not similarities:
            return None, 0.0

        max_sim_index = np.argmax(similarities)
        return target_docs[max_sim_index], similarities[max_sim_index]

    def _get_adjacent_docs(self, doc_id: str) -> List[Document]:
        """Retrieves all documents adjacent to the given doc_id in the graph."""
        adjacent_ids = set()
        for edge in self.graph["edges"]:
            if edge["source"] == doc_id:
                adjacent_ids.add(edge["target"])
            elif edge["target"] == doc_id:
                adjacent_ids.add(edge["source"])
        
        return [self.doc_map[adj_id] for adj_id in adjacent_ids if adj_id in self.doc_map]

    def get_verbose_log(self) -> List[str]:
        return self.verbose_log

    def clear_log(self):
        self.verbose_log = []

__all__ = [
    "SpinRAG",
    "SpinType",
    "Document"
]