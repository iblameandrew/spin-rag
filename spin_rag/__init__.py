"""SpinRAG: an evolving knowledge-graph RAG focused on document restoration.

Public API:

    from spin_rag import SpinRAG, SpinType, Document
"""

from .spin_rag import SpinRAG, SpinType, Document

__version__ = "0.1.0a1"

__all__ = ["SpinRAG", "SpinType", "Document", "__version__"]
