"""Central configuration for IR__Chat.

Every hard limit lives here. In particular MAX_DOCS is defined exactly once so
the "at most 25 documents" rule cannot drift between the loader, the uploader
and the UI.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "IR__Chat"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
KB_DIR = DATA_DIR / "kb"
EVAL_DIR = DATA_DIR / "eval"

# --- Corpus limits -----------------------------------------------------------
# Assignment constraint: the QA system holds a maximum of 25 documents, each of
# which the user can download from the app.
MAX_DOCS = 25

MIN_DOC_CHARS = 120          # anything shorter is noise, not a document
MAX_DOC_CHARS = 200_000      # ~35k words; keeps indexing snappy on Streamlit Cloud
ALLOWED_EXTENSIONS = frozenset({".txt", ".md"})

# --- Passage construction ----------------------------------------------------
# Retrieval works on passages, not whole documents: a 400-word document dilutes
# the term signal, a 3-sentence window keeps it sharp and is small enough to
# show to the user as evidence.
CHUNK_WINDOW_SENTENCES = 3
CHUNK_STRIDE_SENTENCES = 2   # window 3 / stride 2 => 1 sentence of overlap
MIN_PASSAGE_CHARS = 90
