"""Stage 1 self-check for IR__Chat.

Run with:  python3 scripts/stage1_check.py

This is not a unit test suite, it is a proof that the foundation behaves before
anything is built on top of it. It checks four things:

1. the bundled corpus loads and every document has a sane title and size,
2. the 25-document cap actually holds and duplicates are refused,
3. the sentence splitter does not break on abbreviations or initials,
4. every passage still slices exactly out of its source document.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ir_chat import (  # noqa: E402
    MAX_DOCS,
    Corpus,
    CorpusLimitError,
    Document,
    DocumentRejected,
    chunk_corpus,
    load_bundled_corpus,
    verify_offsets,
)
from ir_chat.preprocess import analyze, split_sentences  # noqa: E402

RULE = "-" * 72
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}{(' - ' + detail) if detail else ''}")
    if not condition:
        failures.append(label)


# --- 1. corpus ---------------------------------------------------------------
print(RULE)
print("1. BUNDLED CORPUS")
print(RULE)

corpus, notes = load_bundled_corpus()
for note in notes:
    print(f"  note: {note}")

print(f"\n  {'doc_id':<34}{'words':>6}  title")
for doc in corpus:
    print(f"  {doc.doc_id:<34}{doc.word_count:>6}  {doc.title}")

stats = corpus.stats()
print(
    f"\n  {stats['documents']} documents / {MAX_DOCS} slots, "
    f"{stats['total_words']:,} words, fingerprint {stats['fingerprint']}"
)
check("corpus is non-empty", len(corpus) > 0, f"{len(corpus)} documents")
check("within the 25-document cap", len(corpus) <= MAX_DOCS)
check("every document has a title", all(d.title for d in corpus))
check("every document has a checksum", all(len(d.checksum) == 16 for d in corpus))
check("download payload is non-empty", all(corpus.documents[0].download_payload()))

# --- 2. limits ---------------------------------------------------------------
print(f"\n{RULE}")
print("2. LIMITS AND DEDUPLICATION")
print(RULE)

sandbox = Corpus()


def filler(i: int) -> str:
    return f"Filler document {i} exists only to occupy a slot in the bounded corpus. " * 3


for i in range(MAX_DOCS):
    sandbox.add(Document.from_text(f"filler_{i}.txt", filler(i)))

check("exactly 25 documents fit", len(sandbox) == MAX_DOCS)
check("corpus reports itself full", sandbox.is_full and sandbox.remaining_slots == 0)

try:
    sandbox.add(Document.from_text("one_too_many.txt", filler(99)))
    check("26th document is refused", False)
except CorpusLimitError as exc:
    check("26th document is refused", True, str(exc)[:58] + "\u2026")

sandbox.remove(sandbox.documents[0].doc_id)
check("removing frees a slot", sandbox.remaining_slots == 1)

try:
    sandbox.add(Document.from_text("copy.txt", filler(1)))
    check("byte-identical duplicate is refused", False)
except DocumentRejected:
    check("byte-identical duplicate is refused", True)

try:
    Document.from_text("tiny.txt", "Too short.")
    check("undersized document is refused", False)
except DocumentRejected:
    check("undersized document is refused", True)

# --- 3. sentence splitting ---------------------------------------------------
print(f"\n{RULE}")
print("3. SENTENCE SPLITTING")
print(RULE)

tricky = (
    "WordNet was begun in 1985 by George A. Miller at Princeton. "
    "Dr. Miller worked with e.g. psycholinguists and lexicographers. "
    "Version 3.0 has about 117,000 synsets! Does that count synsets or words?"
)
sentences = split_sentences(tricky)
for s in sentences:
    print(f"  \u2022 {s}")
check("splits into 4 sentences", len(sentences) == 4, f"got {len(sentences)}")
check("keeps the initial 'George A. Miller'", "George A. Miller" in sentences[0])
check("keeps the abbreviation 'Dr.'", sentences[1].startswith("Dr. Miller"))
check("does not split the decimal in '3.0'", "Version 3.0" in sentences[2])

sample_terms = analyze("Who developed the BM25 ranking functions at City University?")
print(f"\n  analyze() -> {sample_terms}")
check("stopwords removed", "the" not in sample_terms and "at" not in sample_terms)
check("plural collapsed by the stemmer", "function" in sample_terms)
check("query word kept as a term", "develop" in sample_terms)

# --- 4. passages -------------------------------------------------------------
print(f"\n{RULE}")
print("4. PASSAGES")
print(RULE)

passages = chunk_corpus(corpus)
per_doc: dict[str, int] = {}
for p in passages:
    per_doc[p.doc_id] = per_doc.get(p.doc_id, 0) + 1

avg = sum(p.char_count for p in passages) / len(passages)
print(f"  {len(passages)} passages, average {avg:.0f} characters")
print(f"  per document: min {min(per_doc.values())}, max {max(per_doc.values())}")
print(f"\n  first passage of {passages[0].doc_id}:")
print(f"    id     {passages[0].passage_id}")
print(f"    span   [{passages[0].start}:{passages[0].end}]")
print(f"    text   {passages[0].snippet(150)}")

problems = verify_offsets(corpus, passages)
check("every document produced passages", len(per_doc) == len(corpus))
check("offsets round-trip to the source text", not problems, f"{len(problems)} problems")
check("passages overlap by design", len(passages) > sum(1 for _ in corpus))

# --- verdict -----------------------------------------------------------------
print(f"\n{RULE}")
if failures:
    print(f"STAGE 1: {len(failures)} CHECK(S) FAILED -> {failures}")
    sys.exit(1)
print("STAGE 1 FOUNDATION OK - ready for the retriever.")
print(RULE)
