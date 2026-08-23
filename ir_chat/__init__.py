"""IR__Chat - an IR-based question answering and chatbot system.

Stage 1: corpus + text foundation.
Stage 2: BM25 retrieval.
Stage 3: factoid answer extraction.
Stage 4: knowledge base and dialogue management.
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
from .extractor import Answer, classify_question, entity_backend, extract_answer
from .dialogue import DialogueManager, Reply, Turn, classify_intent
from .knowledge import Fact, KBAnswer, KnowledgeBase
from .retriever import Retriever, SearchResult, build_retriever

__version__ = "0.4.0"

__all__ = [
    "Answer",
    "Corpus",
    "DialogueManager",
    "CorpusError",
    "CorpusLimitError",
    "Document",
    "Fact",
    "KBAnswer",
    "KnowledgeBase",
    "DocumentRejected",
    "MAX_DOCS",
    "PROJECT_NAME",
    "Passage",
    "Reply",
    "Retriever",
    "SearchResult",
    "Turn",
    "build_retriever",
    "classify_intent",
    "classify_question",
    "chunk_corpus",
    "chunk_document",
    "entity_backend",
    "extract_answer",
    "load_bundled_corpus",
    "verify_offsets",
]
