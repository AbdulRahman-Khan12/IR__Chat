# IR__Chat

A question answering and chatbot system over a bounded document collection.

Built for **AML23702 Advance Natural Language Processing, Lab Exercise 2** —
IR-based factoid answering, knowledge-based QA, and simple dialogue management.

**Constraint:** the corpus holds a maximum of **25 documents**, and every one of
them is downloadable from the interface.

---

## Architecture

```
question
   │
   ├── dialogue manager ──── intent? ──── chit-chat / meta / clarification
   │        (Stage 4)                     (no retrieval needed)
   │
   ├── knowledge base ────── fact hit? ── precise, traceable answer
   │        (Stage 4)
   │
   └── retriever ─────────── passages ─── answer extractor ─── answer + evidence
         (Stage 2)                            (Stage 3)
              ▲
              │
        passages ← chunker ← corpus (≤ 25 docs)
                   (Stage 1)
```

The system tries the cheapest, most precise route first and falls back to
retrieval. Every answer carries the passage it came from, so nothing the app
says is unverifiable.

## Build stages

| Stage | Scope | Status |
|-------|-------|--------|
| 1 | Corpus layer, preprocessing, passage chunking | **done** |
| 2 | Retrieval: TF-IDF and BM25, ranking, query handling | pending |
| 3 | Factoid extraction: answer typing, NER, span scoring | pending |
| 4 | Knowledge base + frame-based dialogue manager | pending |
| 5 | Streamlit interface, evaluation harness, deployment | pending |

## Layout

```
IR__Chat/
├── ir_chat/
│   ├── config.py       all limits, MAX_DOCS lives here and nowhere else
│   ├── preprocess.py   normalise, sentence split (with offsets), tokenize, stem
│   ├── corpus.py       Document, Corpus (capped at 25), download payloads
│   └── chunker.py      sliding sentence windows → Passage objects
├── data/
│   ├── corpus/         12 bundled documents, 13 slots free for uploads
│   ├── kb/             knowledge base triples (Stage 4)
│   └── eval/           question/answer set (Stage 5)
└── scripts/
    └── stage1_check.py self-check for the foundation
```

## Running the Stage 1 check

No dependencies are needed yet — Stage 1 is pure standard library.

```bash
python3 scripts/stage1_check.py
```

It verifies that the corpus loads, that the 26th document is refused, that the
sentence splitter survives abbreviations and initials, and that every passage
still slices exactly out of its source document.

## Design notes

**Why passages instead of documents.** A 200-word document mentions its key
entity once; that signal is diluted across the whole file. A three-sentence
window keeps the signal concentrated and doubles as the evidence snippet shown
to the user.

**Why one sentence of overlap.** Evidence for a factoid frequently straddles a
sentence boundary — the entity is named in one sentence and the date appears in
the next. Window 3 with stride 2 guarantees any two adjacent sentences appear
together in at least one passage.

**Why character offsets.** Passages store `start` and `end` positions into the
document text rather than copies. Stage 3 needs them to locate an answer span
inside the source, and Stage 5 needs them to highlight it. `Document` is a
frozen dataclass so those offsets can never silently go stale.

**Why no NLTK yet.** Streamlit Community Cloud rebuilds the container on every
deploy, and anything calling `nltk.download()` at import time is a startup
failure waiting to happen. NLTK and spaCy arrive in Stage 3, where their models
do work that hand-written rules cannot.
