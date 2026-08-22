"""IR__Chat - an IR-based question answering and chatbot system.

Stage 1: corpus + text foundation.
Stage 2: BM25 retrieval.
"""

from .chunker import Passage, chunk_corpus, chunk_document, verify_offsets
from .config import MAX_DOCS, PROJECT_NAME
from .corpus import (
    Corpus,
    CorpusError,
    CorpusLimitError,
    Document,
    DocumentRejected,
    load_bundled_corpus,
)
from .retriever import Retriever, SearchResult, build_retriever

__version__ = "0.2.0"

__all__ = [
    "Corpus",
    "CorpusError",
    "CorpusLimitError",
    "Document",
    "DocumentRejected",
    "MAX_DOCS",
    "PROJECT_NAME",
    "Passage",
    "Retriever",
    "SearchResult",
    "build_retriever",
    "chunk_corpus",
    "chunk_document",
    "load_bundled_corpus",
    "verify_offsets",
]
