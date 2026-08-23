"""Dialogue management for IR__Chat.

Everything above this module answers one question in isolation. This is what
makes it a conversation.

Three jobs:

    1. Intent      Is this a greeting, a request for help, or a real question?
                   Running retrieval on "hi" is a waste and looks foolish.

    2. Frame       A frame is a task with slots to fill. This system has one
                   task -- answer questions about the corpus -- and one slot,
                   `topic`. GUS (Xerox PARC, 1977) filled slots for flight
                   bookings; the machinery is identical, there is just less of
                   it here.

    3. Routing     Knowledge base first, retrieval second, honest refusal last.
                   A successful lookup is always right, so it is tried first.

The slot is what makes follow-ups work. Once `topic` holds "ELIZA", the turn
"and when?" can be rewritten into "and when ELIZA" before it reaches the
retriever, which has no memory of its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .corpus import Corpus
from .extractor import Answer, extract_answer
from .knowledge import KBAnswer, KnowledgeBase
from .retriever import Retriever, SearchResult

# =============================================================================
# Intents
# =============================================================================

_INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("greeting", re.compile(r"^\s*(hi|hey|hello|yo|good (morning|afternoon|evening))\b")),
    ("farewell", re.compile(r"^\s*(bye|goodbye|see you|that'?s all|exit|quit)\b")),
    ("thanks", re.compile(r"^\s*(thanks|thank you|ta|cheers|nice|great|perfect)\b\s*[.!]?\s*$")),
    ("help", re.compile(r"\b(help|what can you do|how do i use|who are you|what are you)\b")),
    ("documents", re.compile(r"\b(what|which) (documents|docs|topics|sources|files)\b"
                             r"|\blist (the )?(documents|docs|topics|sources)\b"
                             r"|\bwhat do you know about\b")),
]

_PRONOUNS = re.compile(r"\b(it|its|it's|that|this|they|them|their|he|his|she|her)\b", re.I)
_QUESTION_ONLY = re.compile(
    r"^\s*(and\s+|so\s+|ok\s+|okay\s+)?"
    r"(who|what|when|where|why|which|how)"
    r"(\s+(is|are|was|were|did|does|do|about|many|much|long))?\s*\??\s*$",
    re.I,
)


def classify_intent(text: str) -> str:
    lowered = text.lower().strip()
    for name, pattern in _INTENT_RULES:
        if pattern.search(lowered):
            return name
    return "question"


# =============================================================================
# Turns and replies
# =============================================================================


@dataclass(frozen=True)
class Reply:
    """One system turn, with everything the interface needs to render it."""

    text: str
    intent: str
    route: str                     # "knowledge base" | "retrieval" | "none" | conversational
    question: str = ""             # the question actually used, after resolution
    rewritten: bool = False        # was a pronoun or ellipsis resolved?
    answer: Answer | None = None
    fact: KBAnswer | None = None
    results: list[SearchResult] = field(default_factory=list)
    topic: str | None = None

    @property
    def has_evidence(self) -> bool:
        return self.answer is not None or self.fact is not None


@dataclass
class Turn:
    user: str
    reply: Reply


# =============================================================================
# The manager
# =============================================================================


class DialogueManager:
    """Holds the conversation state and decides who answers each turn."""

    def __init__(
        self,
        corpus: Corpus,
        retriever: Retriever | None = None,
        kb: KnowledgeBase | None = None,
    ) -> None:
        self.corpus = corpus
        self.retriever = retriever or Retriever(corpus)
        self.kb = kb or KnowledgeBase()
        self.history: list[Turn] = []
        self.topic: str | None = None      # the single frame slot

    def reset(self) -> None:
        self.history.clear()
        self.topic = None

    # --- slot filling --------------------------------------------------------

    def resolve(self, text: str) -> tuple[str, bool]:
        """Rewrite a context-dependent turn into a standalone question.

        Two cases, both extremely common in real dialogue:

            pronoun    "who built it?"   -> "who built ELIZA?"
            ellipsis   "and when?"       -> "and when ELIZA"

        The rewritten string is not always grammatical, and does not need to be.
        Retrieval strips stopwords anyway; what matters is that the topic word
        reaches the index.
        """
        if not self.topic:
            return text, False

        if _QUESTION_ONLY.match(text):
            return f"{text.strip().rstrip('?')} {self.topic}", True

        if _PRONOUNS.search(text):
            return _PRONOUNS.sub(self.topic, text, count=1), True

        return text, False

    def _remember_topic(self, question: str, fact: KBAnswer | None) -> None:
        """Update the slot.

        The topic is what the question was *about*, not what the answer said.
        After "Who wrote ELIZA?" the topic is ELIZA, not Joseph Weizenbaum,
        so "and when?" asks about ELIZA's date rather than Weizenbaum's.
        """
        if fact is not None:
            self.topic = fact.subject
            return
        subject = self.kb.find_subject(question)
        if subject:
            self.topic = subject

    # --- the main entry point ------------------------------------------------

    def respond(self, text: str) -> Reply:
        text = text.strip()
        if not text:
            return Reply("Ask me something about the documents.", "question", "none")

        intent = classify_intent(text)

        if intent in {"greeting", "farewell", "thanks", "help", "documents"}:
            reply = self._small_talk(intent)
            self.history.append(Turn(text, reply))
            return reply

        question, rewritten = self.resolve(text)

        # --- route 1: knowledge base -----------------------------------------
        fact = self.kb.answer(question)
        if fact is not None:
            self._remember_topic(question, fact)
            reply = Reply(
                text=f"{fact.text}.",
                intent=intent,
                route="knowledge base",
                question=question,
                rewritten=rewritten,
                fact=fact,
                topic=self.topic,
            )
            self.history.append(Turn(text, reply))
            return reply

        # --- route 2: retrieval + extraction ---------------------------------
        results = self.retriever.search(question, top_k=5)
        answer = extract_answer(question, results)
        self._remember_topic(question, None)

        if answer.found:
            reply = Reply(
                text=answer.text,
                intent=intent,
                route="retrieval",
                question=question,
                rewritten=rewritten,
                answer=answer,
                results=results,
                topic=self.topic,
            )
        else:
            reply = Reply(
                text=self._refusal(question),
                intent=intent,
                route="none",
                question=question,
                rewritten=rewritten,
                results=results,
                topic=self.topic,
            )

        self.history.append(Turn(text, reply))
        return reply

    # --- conversational turns ------------------------------------------------

    def _small_talk(self, intent: str) -> Reply:
        if intent == "greeting":
            return Reply(
                f"Hello. I answer questions from {len(self.corpus)} documents on "
                "information retrieval and NLP. Ask me anything about them.",
                intent, "conversation",
            )
        if intent == "farewell":
            return Reply("Goodbye.", intent, "conversation")
        if intent == "thanks":
            return Reply("Happy to help. Anything else?", intent, "conversation")
        if intent == "documents":
            titles = "; ".join(d.title for d in self.corpus)
            return Reply(
                f"I have {len(self.corpus)} documents: {titles}.",
                intent, "conversation",
            )
        return Reply(
            "Ask a factoid question and I will find the answer and show you the "
            "sentence it came from. Questions starting who, when, where, how many "
            "get a short answer; why and what-is questions get the relevant "
            "sentence. Follow-ups work too, so you can ask \u201cand when?\u201d after "
            "\u201cwho wrote ELIZA?\u201d.",
            intent, "conversation",
        )

    def _refusal(self, question: str) -> str:
        """Explain the failure instead of apologising vaguely."""
        _, unknown = self.retriever.analyze_query(question)
        if unknown:
            missing = ", ".join(f"\u201c{t}\u201d" for t in unknown[:3])
            return (
                f"Nothing in the {len(self.corpus)} documents mentions {missing}. "
                "Try asking about retrieval, chatbots, embeddings, or evaluation."
            )
        return (
            "I found related passages but nothing that answers that directly. "
            "Try rephrasing, or ask about a specific system such as BM25, ELIZA "
            "or WordNet."
        )

    # --- for the interface ---------------------------------------------------

    def suggestions(self) -> list[str]:
        return [
            "Who wrote ELIZA?",
            "When was WordNet begun?",
            "Which company released word2vec?",
            "What is a synset?",
            "How many synsets does WordNet contain?",
            "Why does BM25 normalise for length?",
        ]

    def stats(self) -> dict[str, object]:
        return {
            "turns": len(self.history),
            "topic": self.topic or "-",
            **self.kb.stats(),
            **self.retriever.stats(),
        }
