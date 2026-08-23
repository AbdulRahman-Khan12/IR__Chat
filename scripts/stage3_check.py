"""Stage 3 self-check for IR__Chat.

Run with:  python3 scripts/stage3_check.py

Measures end-to-end factoid accuracy on a hand-written probe set, shows the
score breakdown behind one answer, and confirms the system degrades honestly
when it has nothing to say.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ir_chat import load_bundled_corpus  # noqa: E402
from ir_chat.extractor import (  # noqa: E402
    classify_question,
    entity_backend,
    extract_answer,
    find_entities,
)
from ir_chat.retriever import Retriever  # noqa: E402

RULE = "-" * 72
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    if not condition:
        failures.append(label)


corpus, _ = load_bundled_corpus()
retriever = Retriever(corpus)


def answer(question: str):
    return extract_answer(question, retriever.search(question, top_k=5))


# --- 1. question typing ------------------------------------------------------
print(RULE)
print("1. WHAT KIND OF ANSWER IS BEING ASKED FOR")
print(RULE)

typing_probes = [
    ("Who wrote ELIZA?", "PERSON"),
    ("When was WordNet begun?", "DATE"),
    ("In what year did TREC start?", "DATE"),
    ("Where was PARRY built?", "PLACE"),
    ("How many synsets are there?", "NUMBER"),
    ("Which company released word2vec?", "ORG"),
    ("What is a synset?", "DEFINITION"),
    ("Why does BM25 normalise length?", "EXPLANATION"),
]
typed = 0
for question, expected in typing_probes:
    got = classify_question(question).name
    typed += got == expected
    flag = " " if got == expected else "!"
    print(f"  {flag} {expected:<12} <- {question}")
check("question types are correct", typed == len(typing_probes), f"{typed}/{len(typing_probes)}")

print(f"\n  entity recogniser in use: {entity_backend()}")
sample = "BM25 was refined at TREC, organised by NIST since 1992."
print(f"  entities in a sample sentence: {[(e.text, e.label) for e in find_entities(sample)]}")
check("entity recogniser returns something", bool(find_entities(sample)))

# --- 2. factoid accuracy -----------------------------------------------------
print(f"\n{RULE}")
print("2. FACTOID ACCURACY")
print(RULE)

probes = [
    ("Who developed BM25?", "Stephen Robertson"),
    ("Who wrote ELIZA?", "Joseph Weizenbaum"),
    ("Who built PARRY?", "Kenneth Colby"),
    ("Who published the Chinese Room argument?", "John Searle"),
    ("Who introduced inverse document frequency?", "Karen Sparck Jones"),
    ("When was WordNet begun?", "1985"),
    ("When did Watson win Jeopardy?", "February 2011"),
    ("When was SQuAD released?", "2016"),
    ("When was Wikidata launched?", "October 2012"),
    ("Where was PARRY built?", "Stanford University"),
    ("Which company released word2vec?", "Google"),
    ("How many synsets does WordNet contain?", "seventeen thousand"),
]

correct = 0
print(f"  {'conf':<7}{'expected':<24}{'got':<34}ok")
for question, expected in probes:
    a = answer(question)
    ok = expected.lower() in a.text.lower()
    correct += ok
    print(f"  {a.confidence:<7}{expected:<24}{a.text[:32]:<34}{'yes' if ok else 'NO'}")

accuracy = correct / len(probes)
print(f"\n  exact-ish match: {correct}/{len(probes)} = {accuracy:.0%}")
check("factoid accuracy is at least 80%", accuracy >= 0.8, f"{accuracy:.0%}")
check("every answer names its source document", all(answer(q).doc_id for q, _ in probes))

# --- 3. why that answer ------------------------------------------------------
print(f"\n{RULE}")
print("3. WHY THAT ANSWER")
print(RULE)

probe = "How many synsets does WordNet contain?"
a = answer(probe)
print(f"  Q: {probe}")
print(f"  A: {a.text}   [{a.label}, confidence {a.confidence}]")
print(f"  from: {a.doc_title} ({a.passage_id})\n")
for line in a.explain():
    print(f"    {line}")
print(f"\n  supporting sentence:\n    {a.sentence}")
print(f"\n  runner-up candidates: {a.alternatives}")

check("score parts sum to the confidence", abs(sum(a.parts.values()) - a.confidence) < 0.01)
check("the version number lost to the real count", "3.0" not in a.text)
check("offsets point back into the document",
      corpus.get(a.doc_id).text[a.doc_start:a.doc_end] == a.text)

# --- 4. sentence fallback ----------------------------------------------------
print(f"\n{RULE}")
print("4. WHEN A SPAN IS THE WRONG SHAPE OF ANSWER")
print(RULE)

for question in ["What is a synset?", "Why does BM25 use length normalisation?",
                 "How does self-attention work?"]:
    a = answer(question)
    print(f"  Q: {question}")
    print(f"     [{a.question_type}/{a.kind}] {a.text[:96]}")

check("definitional questions answer with a sentence", answer("What is a synset?").kind == "sentence")
check("why-questions answer with a sentence",
      answer("Why does BM25 use length normalisation?").kind == "sentence")

# --- 5. honest failure -------------------------------------------------------
print(f"\n{RULE}")
print("5. QUESTIONS THE CORPUS CANNOT ANSWER")
print(RULE)

for question in ["Who won the 2022 World Cup?", "What is the capital of Peru?"]:
    a = answer(question)
    print(f"  Q: {question}\n     kind={a.kind} conf={a.confidence} text={a.text[:60]!r}")

check("out-of-corpus question returns nothing",
      answer("Who won the 2022 World Cup?").kind == "none")

# --- verdict -----------------------------------------------------------------
print(f"\n{RULE}")
if failures:
    print(f"STAGE 3: {len(failures)} CHECK(S) FAILED -> {failures}")
    sys.exit(1)
print("STAGE 3 EXTRACTION OK - ready for the knowledge base and dialogue.")
print(RULE)
