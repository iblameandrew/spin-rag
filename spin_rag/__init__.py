"""SpinRAG: an evolving knowledge-graph RAG focused on document restoration.

Public API::

    from spin_rag import SpinRAG, SpinType, Document, BACKEND_OPENROUTER, BACKEND_LLAMACPP

Supported model backends (both speak the OpenAI HTTP protocol):

- ``BACKEND_OPENROUTER`` (default) — OpenRouter's hosted model catalog.
- ``BACKEND_LLAMACPP`` — a self-hosted `llama.cpp` server.

See the module docstring in :mod:`spin_rag.spin_rag` for full details.
"""

from .spin_rag import (
    BACKEND_LLAMACPP,
    BACKEND_OPENROUTER,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLAMACPP_BASE_URL,
    DEFAULT_LLAMACPP_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    Document,
    SpinRAG,
    SpinType,
    parse_spin,
)
from .spin_rag import (
    _clean_llm_output as clean_llm_output,
)
from .spin_rag import (
    _cosine_similarity as cosine_similarity,
)

__version__ = "0.2.0"

__all__ = [
    "SpinRAG",
    "SpinType",
    "Document",
    "BACKEND_OPENROUTER",
    "BACKEND_LLAMACPP",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_LLAMACPP_BASE_URL",
    "DEFAULT_LLAMACPP_MODEL",
    "DEFAULT_OPENROUTER_BASE_URL",
    "clean_llm_output",
    "cosine_similarity",
    "parse_spin",
    "__version__",
]
