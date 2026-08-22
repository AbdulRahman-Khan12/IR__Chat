"""Stage 2 self-check for IR__Chat.

Run with:  python3 scripts/stage2_check.py

Checks that the index is built correctly, that known questions retrieve the
right document, that the ranking can be explained term by term, and that an
out-of-vocabulary query fails honestly instead of returning noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ir_chat import load_bundled_corpus  # noqa: E402
from ir_chat.retriever import Retriever  # noqa: E402

RULE = "-" * 72
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    if not condition:
        failures.append(label)


corpus, _ = load_bundled_corpus()
retriever = Retriever(corpus)

# --- 1. the index ------------------------------------------------------------
print(RULE)
print("1. INDEX")
print(RULE)

stats = retriever.stats()
for key, value in stats.items():
    print(f"  {key:<20} {value}")

check("index covers every passage", stats["passages"] == len(retriever.passages))
check("vocabulary is non-trivial", stats["vocabulary"] > 300)
check("average passage length is sane", 15 < stats["avg_passage_terms"] < 60)

# --- 2. ranking --------------------------------------------------------------
print(f"\n{RULE}")
print("2. RANKING")
print(RULE)

expectations = [
    ("Who developed BM25?", "04-bm25-ranking"),
    ("When did Watson win Jeopardy?", "05-question-answering-systems"),
    ("What is a synset?", "12-wordnet-and-lexical-semantics"),
    ("Who wrote ELIZA?", "02-eliza-and-early-chatbots"),
    ("What does self-attention do?", "08-transformers"),
    ("How is mean reciprocal rank calculated?", "11-evaluation-metrics"),
    ("Which language queries Neo4j?", "09-knowledge-graphs"),
]

hits = 0
for question, expected_doc in expectations:
    results = retriever.search(question, top_k=3)
    top = results[0] if results else None
    ok = top is not None and top.doc_id == expected_doc
    hits += ok
    print(f"\n  Q: {question}")
    for r in results:
        marker = "->" if r.doc_id == expected_doc else "  "
        print(f"   {marker} {r.rank}. {r.score:5.2f}  {r.doc_title[:40]:<40} {r.matched_terms}")

print()
check(
    "top-1 document is correct for every probe",
    hits == len(expectations),
    f"{hits}/{len(expectations)}",
)

# --- 3. explanation ----------------------------------------------------------
print(f"\n{RULE}")
print("3. WHY DID IT RANK THERE")
print(RULE)

probe = "Who developed the BM25 ranking function?"
best = retriever.search(probe, top_k=1)[0]
print(f"  Q: {probe}")
print(f"  top passage: {best.passage.passage_id}  (total {best.score:.3f})\n")
print(f"  {'term':<14}{'tf':>4}{'df':>5}{'idf':>8}{'score':>9}")
for row in retriever.explain(best):
    print(f"  {row['term']:<14}{row['tf']:>4}{row['df']:>5}{row['idf']:>8}{row['score']:>9}")

rows = retriever.explain(best)
check("contributions sum to the total score", abs(sum(r["score"] for r in rows) - best.score) < 0.01)
check("rarer term outscores the common one", rows[0]["df"] <= rows[-1]["df"])

# --- 4. document-level view --------------------------------------------------
print(f"\n{RULE}")
print("4. DOCUMENT-LEVEL RANKING")
print(RULE)

question = "How do we measure whether the retrieved passages are any good?"
print(f"  Q: {question}\n")
docs = retriever.search_documents(question, top_k=3)
for r in docs:
    print(f"   {r.rank}. {r.score:5.2f}  {r.doc_title}")
    print(f"      {r.snippet(110)}")

check("one result per document", len({r.doc_id for r in docs}) == len(docs))

# --- 5. BM25 vs TF-IDF -------------------------------------------------------
print(f"\n{RULE}")
print("5. BM25 vs TF-IDF ON THE SAME INDEX")
print(RULE)

comparison = "What is the vector space model used for in retrieval?"
print(f"  Q: {comparison}\n")
for scorer in ("bm25", "tfidf"):
    print(f"  {scorer}:")
    for r in retriever.search(comparison, top_k=3, scorer=scorer):
        print(f"     {r.rank}. {r.score:6.3f}  {r.doc_title[:44]:<44} {r.passage.passage_id}")

bm25_top = retriever.search(comparison, top_k=3)
tfidf_top = retriever.search(comparison, top_k=3, scorer="tfidf")
check("both scorers return results", bool(bm25_top) and bool(tfidf_top))
check("both agree on the top document", bm25_top[0].doc_id == tfidf_top[0].doc_id)

# --- 6. honest failure -------------------------------------------------------
print(f"\n{RULE}")
print("6. QUERIES THE CORPUS CANNOT ANSWER")
print(RULE)

for bad in ["quantum cryptography in Antarctica", "the of and a"]:
    known, unknown = retriever.analyze_query(bad)
    results = retriever.search(bad)
    print(f"  Q: {bad}")
    print(f"     known {known}  unknown {unknown}  results {len(results)}")

known, unknown = retriever.analyze_query("quantum cryptography in Antarctica")
check("unseen terms are reported, not swallowed", len(unknown) == 3 and not known)
check("unanswerable query returns nothing", retriever.search("quantum cryptography") == [])
check("stopword-only query returns nothing", retriever.search("the of and a") == [])

# --- verdict -----------------------------------------------------------------
print(f"\n{RULE}")
if failures:
    print(f"STAGE 2: {len(failures)} CHECK(S) FAILED -> {failures}")
    sys.exit(1)
print("STAGE 2 RETRIEVAL OK - ready for answer extraction.")
print(RULE)
