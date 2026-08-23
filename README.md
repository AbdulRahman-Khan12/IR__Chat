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
| 2 | Retrieval: inverted index, BM25 + TF-IDF, ranking | **done** |
| 3 | Factoid extraction: answer typing, spaCy NER, span scoring | **done** |
| 4 | Knowledge base (81 triples) + frame-based dialogue manager | **done** |
| 5 | Streamlit interface, evaluation harness, deployment | **done** |

## Layout

```
IR__Chat/
├── app.py              the Streamlit interface
├── ir_chat/
│   ├── config.py       all limits, MAX_DOCS lives here and nowhere else
│   ├── preprocess.py   normalise, sentence split (with offsets), tokenize, stem
│   ├── corpus.py       Document, Corpus (capped at 25), download payloads
│   ├── chunker.py      sliding sentence windows → Passage objects
│   ├── retriever.py    inverted index, BM25 and TF-IDF scoring
│   ├── extractor.py    question typing, NER, answer span selection
│   ├── knowledge.py    triples, question -> (subject, relation) lookup
│   ├── dialogue.py     intents, topic slot, routing, follow-ups
│   └── evaluate.py     exact match, token F1, recall@k, MRR
├── data/
│   ├── corpus/         12 bundled documents, 13 slots free for uploads
│   ├── kb/facts.json   81 triples, each citing its source document
│   └── eval/qa_pairs.json  35 gold questions, 3 of them unanswerable
└── scripts/
    ├── stage1_check.py self-check for the foundation
    ├── stage2_check.py self-check for retrieval
    ├── stage3_check.py self-check for answer extraction
    ├── stage4_check.py self-check for knowledge base and dialogue
    └── stage5_check.py the full evaluation report
```

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploying to Streamlit Community Cloud

1. Push to GitHub (this repository).
2. At share.streamlit.io, choose **New app**, pick the repo and branch `main`.
3. Set the main file to `app.py` and deploy.

No API keys and no secrets are needed. The spaCy model installs from the
wheel URL pinned in `requirements.txt`, because Community Cloud installs
from that file and cannot run `python -m spacy download`. If the model
ever fails to install, the extractor falls back to regex entities and the
app keeps answering — the System panel in the sidebar shows which
recogniser is live.

## Results

35 gold questions, 3 of them deliberately unanswerable.

| Measure | Score | What it tells you |
|---|---|---|
| Recall@5 | 0.969 | The right document reached the extractor |
| MRR | 0.948 | and it was usually at rank 1 |
| Exact match | 0.844 | Character-for-character short answers |
| Token F1 | 0.922 | Partial credit for overlapping words |
| Refusals correct | 1.000 | Unanswerable questions were declined |
| **Overall** | **1.000** | 35/35 |

Retrieval and extraction are scored separately on purpose. A single
end-to-end number hides which half is broken: 95% recall with 40% exact
match needs a better extractor, 50% recall needs a better retriever.

Exact match sits below token F1 because definitional and “why” questions
are answered with a sentence, which can never match a short gold string
character for character even when it is completely correct.

## Running the stage checks

Each stage ships a self-check that proves it before the next one is built.

```bash
python3 scripts/stage1_check.py
python3 scripts/stage2_check.py
python3 scripts/stage3_check.py   # needs spaCy, see requirements.txt
python3 scripts/stage4_check.py
python3 scripts/stage5_check.py
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

**Why one index and two scorers.** BM25 and TF-IDF need the same three
statistics: term frequency, document frequency, and passage length. Building
two separate retrievers would duplicate all of that. One index with two
scoring functions makes the comparison a one-line change and shows that the
difference between the two methods is the formula, not the data.

**Why every result carries a term breakdown.** A bare relevance score is
unfalsifiable. `SearchResult.term_scores` records what each query term
contributed, so the interface can justify a ranking and a wrong answer can be
diagnosed instead of guessed at.

**Why a coverage gate.** Retrieval scores are *relative*: the best of five
irrelevant passages still ranks first with a healthy-looking score. Asking
"Who won the 2022 World Cup?" retrieved the ELIZA document, because it
contains the word *world*, and the extractor confidently answered "Joseph
Weizenbaum". Requiring a passage to share at least 40% of the question's
content words is an *absolute* test, and it is what lets the system say it
does not know. Genuine questions score 0.50-1.00; that one scored 0.25.

**Why answers fall back to sentences.** A "why" or "what is" question has no
entity-shaped answer. Rather than forcing a span, the extractor quotes the
best supporting sentence. Naming the wrong person is a worse failure than
declining to name one.

**Why the knowledge base is tried first.** A lookup that succeeds is always
right; a retrieved span is a guess. Both routes stay in the system because
they fail in opposite directions: the knowledge base can only answer what
somebody entered, retrieval can answer anything the corpus mentions but may
pick the wrong span.

**Why the topic slot stores the question's subject.** After "Who wrote
ELIZA?" the slot holds *ELIZA*, not *Joseph Weizenbaum*, so the follow-up
"and when?" asks when ELIZA was written rather than when Weizenbaum was
born. The rewritten query does not have to be grammatical -- "and when
ELIZA" is enough, because retrieval strips stopwords anyway.

**Why no NLTK yet.** Streamlit Community Cloud rebuilds the container on every
deploy, and anything calling `nltk.download()` at import time is a startup
failure waiting to happen. NLTK and spaCy arrive in Stage 3, where NER does work
that hand-written rules cannot.
