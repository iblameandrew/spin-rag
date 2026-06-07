"""SpinRAG: an evolving knowledge-graph RAG focused on document restoration.

Public API::

    from spin_rag import SpinRAG, SpinType, Document, BACKEND_OPENROUTER, BACKEND_LLAMACPP

Supported model backends (both speak the OpenAI HTTP protocol):

- ``BACKEND_OPENROUTER`` (default) — OpenRouter's hosted model catalog.
- ``BACKEND_LLAMACPP`` — a self-hosted `llama.cpp` server.

See the module docstring in :mod:`spin_rag.spin_rag` for full details.
"""

from .spin_rag import (
    SpinRAG,
    SpinType,
    Document,
    BACKEND_OPENROUTER,
    BACKEND_LLAMACPP,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLAMACPP_BASE_URL,
)

__version__ = "0.1.0"

__all__ = [
    "SpinRAG",
    "SpinType",
    "Document",
    "BACKEND_OPENROUTER",
    "BACKEND_LLAMACPP",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_LLAMACPP_BASE_URL",
    "__version__",
]
