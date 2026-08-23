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

# --- Retrieval ---------------------------------------------------------------
# BM25's two knobs. k1 controls how fast term frequency saturates: a passage
# mentioning "bm25" ten times is not ten times more relevant than one mentioning
# it once. b controls length normalisation: 0 ignores length entirely, 1
# penalises long passages fully. These are the standard defaults.
BM25_K1 = 1.5
BM25_B = 0.75

DEFAULT_TOP_K = 5

# --- Answer extraction -------------------------------------------------------
# A candidate answer is scored on three positive signals and one penalty. The
# weights sum to 1.0, so a raw score is already a usable confidence.
W_EVIDENCE = 0.30   # how relevant was the passage this came from
W_TYPE = 0.30       # is this the kind of thing the question asked for
W_CONTEXT = 0.40    # does the sentence around it address the question
W_REPEAT = 0.35     # penalty: the answer just echoes words from the question

# Candidates scoring below this are not worth showing as a factoid answer;
# the system falls back to quoting the best sentence instead.
MIN_ANSWER_CONFIDENCE = 0.35

# A passage must share at least this fraction of the question's content words
# before any answer is taken from it. Without this gate the extractor happily
# answers "Who won the 2022 World Cup?" from a passage that merely contains
# the word "world", because retrieval scores are relative: the best of five
# bad passages still ranks first.
MIN_QUESTION_COVERAGE = 0.4

# How many retrieved passages the extractor scans for candidates.
EXTRACT_FROM_TOP_N = 4
