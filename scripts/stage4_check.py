"""Stage 4 self-check for IR__Chat.

Run with:  python3 scripts/stage4_check.py

Verifies the knowledge base answers by lookup, that the dialogue manager routes
to it before falling back to retrieval, and that follow-up turns carrying a
pronoun or nothing but a question word still reach the right fact.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ir_chat import load_bundled_corpus  # noqa: E402
from ir_chat.dialogue import DialogueManager, classify_intent  # noqa: E402
from ir_chat.knowledge import KnowledgeBase  # noqa: E402

RULE = "-" * 72
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    if not condition:
        failures.append(label)


corpus, _ = load_bundled_corpus()
kb = KnowledgeBase()
manager = DialogueManager(corpus, kb=kb)

# --- 1. the knowledge base ---------------------------------------------------
print(RULE)
print("1. KNOWLEDGE BASE")
print(RULE)

for key, value in kb.stats().items():
    print(f"  {key:<20} {value}")

print(f"\n  everything it knows about WordNet:")
for fact in kb.facts_about("WordNet"):
    print(f"    {fact.as_triple()}")

check("facts loaded", len(kb) > 50, f"{len(kb)} facts")
check("every fact cites a document", all(f.doc_id for f in kb.facts))
check("every fact uses a declared relation", all(f.relation in kb.relations for f in kb.facts))
check("every cited document exists", all(corpus.get(f.doc_id) for f in kb.facts))

# --- 2. question -> structured lookup ----------------------------------------
print(f"\n{RULE}")
print("2. QUESTION PARSED INTO A LOOKUP")
print(RULE)

lookups = [
    ("Who wrote ELIZA?", "Joseph Weizenbaum"),
    ("When was WordNet begun?", "1985"),
    ("Which university built PARRY?", "Stanford"),
    ("Who created the Lesk algorithm?", "Michael Lesk"),
    ("When was Wikidata launched?", "October 2012"),
    ("Who introduced inverse document frequency?", "Karen Sparck Jones"),
    ("Which journal published IDF?", "Journal of Documentation"),
    ("Which company is behind GPT?", "OpenAI"),
]
hits = 0
for question, expected in lookups:
    result = kb.answer(question)
    ok = result is not None and expected.lower() in result.text.lower()
    hits += ok
    got = result.text if result else "(no match)"
    print(f"  {'ok ' if ok else 'NO '} {question:<44} {got[:30]}")
check("knowledge base lookups are correct", hits == len(lookups), f"{hits}/{len(lookups)}")

print(f"\n  parse of a single question:")
for line in kb.answer("Who wrote ELIZA?").explain():
    print(f"    {line}")

check("longest subject wins", kb.find_subject("when was the Google Knowledge Graph announced")
      == "Google Knowledge Graph")
check("alias resolves to canonical subject", kb.find_subject("who created IDF") == "inverse document frequency")
check("unknown subject returns nothing", kb.answer("Who wrote Hamlet?") is None)

# --- 3. intents --------------------------------------------------------------
print(f"\n{RULE}")
print("3. INTENT ROUTING")
print(RULE)

intents = [
    ("hello there", "greeting"),
    ("thanks", "thanks"),
    ("bye", "farewell"),
    ("what can you do?", "help"),
    ("what documents do you have?", "documents"),
    ("Who wrote ELIZA?", "question"),
]
correct = 0
for text, expected in intents:
    got = classify_intent(text)
    correct += got == expected
    print(f"  {'ok ' if got == expected else 'NO '} {expected:<12} <- {text}")
check("intents are correct", correct == len(intents), f"{correct}/{len(intents)}")

# --- 4. follow-ups -----------------------------------------------------------
print(f"\n{RULE}")
print("4. FOLLOW-UPS (THE FRAME SLOT AT WORK)")
print(RULE)

manager.reset()
conversation = [
    "Who wrote ELIZA?",
    "and when?",
    "where was it built?",
    "Who developed BM25?",
    "what are its typical parameter values?",
]
for line in conversation:
    reply = manager.respond(line)
    arrow = f"  (resolved -> {reply.question})" if reply.rewritten else ""
    print(f"  USER  {line}")
    print(f"  BOT   [{reply.route}] {reply.text[:70]}{arrow}")
    print(f"        topic slot = {reply.topic}\n")

manager.reset()
manager.respond("Who wrote ELIZA?")
check("topic slot holds the question subject, not the answer", manager.topic == "ELIZA")
check("bare question word resolves", manager.respond("and when?").text.startswith("1966"))
check("pronoun resolves", "Massachusetts" in manager.respond("where was it built?").text)

manager.reset()
check("no topic yet means no rewriting", manager.respond("and when?").route == "none")

# --- 5. routing precedence ---------------------------------------------------
print(f"{RULE}")
print("5. WHICH ROUTE ANSWERED")
print(RULE)

manager.reset()
routing = [
    ("Who wrote ELIZA?", "knowledge base"),
    ("What is a synset?", "retrieval"),
    ("Why does BM25 normalise for length?", "retrieval"),
    ("Who won the 2022 World Cup?", "none"),
    ("hello", "conversation"),
]
correct = 0
for text, expected in routing:
    reply = manager.respond(text)
    correct += reply.route == expected
    print(f"  {'ok ' if reply.route == expected else 'NO '} {expected:<16} <- {text}")
check("routes are correct", correct == len(routing), f"{correct}/{len(routing)}")

manager.reset()
refusal = manager.respond("Who won the 2022 World Cup?")
print(f"\n  refusal message:\n    {refusal.text}")
check("refusal names the missing words", "2022" in refusal.text)
check("conversation history is recorded", len(manager.history) == 1)

# --- verdict -----------------------------------------------------------------
print(f"\n{RULE}")
if failures:
    print(f"STAGE 4: {len(failures)} CHECK(S) FAILED -> {failures}")
    sys.exit(1)
print("STAGE 4 KNOWLEDGE + DIALOGUE OK - ready for the interface.")
print(RULE)
