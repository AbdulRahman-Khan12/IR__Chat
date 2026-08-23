"""Answer extraction for IR__Chat.

Retrieval hands over passages. This module turns a passage into an answer.

The pipeline is three steps, and each one is a question you could answer by
hand:

    1. What kind of thing is being asked for?      classify_question()
       "Who wrote ELIZA?" wants a PERSON, not a date.

    2. What things of that kind are in the passages?   find_entities()
       spaCy's named entity recogniser, with a regex fallback so the app still
       works if the model is unavailable.

    3. Which of those is the answer?                   extract_answer()
       Score every candidate on evidence, type match and context, minus a
       penalty for simply echoing the question back.

When no candidate is convincing enough, the extractor quotes the best
supporting sentence rather than inventing a span. Saying "here is the relevant
sentence" is a useful answer; guessing a wrong name is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import (
    EXTRACT_FROM_TOP_N,
    MIN_ANSWER_CONFIDENCE,
    MIN_QUESTION_COVERAGE,
    W_CONTEXT,
    W_EVIDENCE,
    W_REPEAT,
    W_TYPE,
)
from .preprocess import analyze, simple_stem, split_sentences_with_spans
from .retriever import SearchResult

# =============================================================================
# 1. What kind of thing is being asked for?
# =============================================================================


@dataclass(frozen=True)
class QuestionType:
    name: str
    labels: frozenset[str]      # spaCy entity labels that would answer it
    wants_span: bool = True     # False => answer with a sentence, not an entity


# Order matters. "How many" must be tested before "how", and "what year" before
# the generic "what", otherwise the broader pattern swallows the narrower one.
_QUESTION_RULES: list[tuple[re.Pattern[str], QuestionType]] = [
    (re.compile(r"\bhow (many|much|large|big|long is)\b"),
     QuestionType("NUMBER", frozenset({"CARDINAL", "QUANTITY", "PERCENT", "MONEY", "ORDINAL"}))),
    (re.compile(r"\b(who|whom|whose)\b"),
     QuestionType("PERSON", frozenset({"PERSON"}))),
    (re.compile(r"\b(when|what year|which year|in what year)\b"),
     QuestionType("DATE", frozenset({"DATE"}))),
    (re.compile(r"\bwhere\b"),
     QuestionType("PLACE", frozenset({"GPE", "LOC", "FAC", "ORG"}))),
    (re.compile(r"\b(which|what) (company|organisation|organization|university|"
                r"institution|lab|team|group)\b"),
     QuestionType("ORG", frozenset({"ORG"}))),
    (re.compile(r"\b(what|which) (is|are|was|were) (a|an|the)?\s*\w+\??$"),
     QuestionType("DEFINITION", frozenset(), wants_span=False)),
    (re.compile(r"\b(define|what does .+ mean|what is meant by)\b"),
     QuestionType("DEFINITION", frozenset(), wants_span=False)),
    (re.compile(r"\b(why|how does|how do|how is|how are|explain)\b"),
     QuestionType("EXPLANATION", frozenset(), wants_span=False)),
    (re.compile(r"\b(which|what) (language|model|algorithm|method|dataset|"
                r"paper|book|system|tool)\b"),
     QuestionType("TITLE", frozenset({"WORK_OF_ART", "PRODUCT", "EVENT", "ORG", "LANGUAGE"}))),
]

# Anything unmatched: accept any entity, but at reduced type confidence.
_FALLBACK_TYPE = QuestionType("OTHER", frozenset())


def classify_question(question: str) -> QuestionType:
    """Map a question to the kind of answer it expects.

    This is the classic 'answer type detection' step of an IR-based QA system.
    It is deliberately a rule table rather than a classifier: with 25 documents
    there is no training data, and rules are inspectable when they misfire.
    """
    text = question.lower().strip()
    for pattern, qtype in _QUESTION_RULES:
        if pattern.search(text):
            return qtype
    return _FALLBACK_TYPE


# =============================================================================
# 2. What things of that kind are in the passages?
# =============================================================================


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    start: int   # offset within the passage text
    end: int


_nlp = None
_spacy_state = "unloaded"   # "spacy" | "regex" | "unloaded"


def _load_spacy():
    """Load spaCy once, lazily, and never crash if it is missing.

    Streamlit Cloud can fail to install the model. Falling back to regex keeps
    the app answering questions with reduced accuracy instead of showing a
    stack trace.
    """
    global _nlp, _spacy_state
    if _spacy_state != "unloaded":
        return _nlp
    try:
        import spacy

        _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])
        _spacy_state = "spacy"
    except Exception:
        _nlp = None
        _spacy_state = "regex"
    return _nlp


def entity_backend() -> str:
    """Which recogniser is actually running: 'spacy' or 'regex'."""
    _load_spacy()
    return _spacy_state


# Fallback patterns, used only when spaCy is unavailable.
_RE_YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_RE_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_RE_PROPER = re.compile(r"\b[A-Z][a-zA-Z0-9.\-]*(?:\s+[A-Z][a-zA-Z0-9.\-]*)*\b")


def _regex_entities(text: str) -> list[Entity]:
    found: list[Entity] = []
    for match in _RE_YEAR.finditer(text):
        found.append(Entity(match.group(), "DATE", match.start(), match.end()))
    for match in _RE_NUMBER.finditer(text):
        if not any(e.start <= match.start() < e.end for e in found):
            found.append(Entity(match.group(), "CARDINAL", match.start(), match.end()))
    for match in _RE_PROPER.finditer(text):
        phrase = match.group()
        # A lone capitalised word at the start of a sentence is usually just
        # capitalisation, not a name.
        if " " not in phrase and (match.start() == 0 or text[match.start() - 2] in ".!?"):
            continue
        found.append(Entity(phrase, "PROPN", match.start(), match.end()))
    return found


def find_entities(text: str) -> list[Entity]:
    """Named entities in a passage, from spaCy when available."""
    nlp = _load_spacy()
    if nlp is None:
        return _regex_entities(text)
    return [
        Entity(ent.text, ent.label_, ent.start_char, ent.end_char)
        for ent in nlp(text).ents
    ]


# PROPN is the regex fallback's catch-all. Treat it as a partial match for any
# name-like question so the fallback still ranks sensibly.
_LOOSE_LABELS = frozenset({"PROPN"})


def _type_match(label: str, qtype: QuestionType) -> float:
    if not qtype.labels:            # OTHER: any entity is equally plausible
        return 0.4
    if label in qtype.labels:
        return 1.0
    if label in _LOOSE_LABELS:
        return 0.5
    return 0.0


# =============================================================================
# 3. Which candidate is the answer?
# =============================================================================


@dataclass(frozen=True)
class Answer:
    """What the system decided, and everything needed to check it."""

    text: str
    kind: str                       # "span" | "sentence" | "none"
    confidence: float
    question_type: str
    label: str = ""                 # entity label, for span answers
    sentence: str = ""              # the sentence the answer sits in
    doc_id: str = ""
    doc_title: str = ""
    doc_start: int = 0              # offsets into Document.text, for highlighting
    doc_end: int = 0
    passage_id: str = ""
    parts: dict[str, float] = field(default_factory=dict)
    alternatives: list[tuple[str, float]] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.kind != "none"

    def explain(self) -> list[str]:
        """One readable line per scoring component."""
        readable = {
            "evidence": "passage relevance",
            "type": f"matches expected type {self.question_type}",
            "context": "sentence overlaps the question",
            "repeat": "penalty for echoing the question",
        }
        return [
            f"{readable.get(k, k):<38} {v:+.3f}"
            for k, v in self.parts.items()
        ]


def _sentence_around(text: str, position: int) -> tuple[str, int, int]:
    for start, end in split_sentences_with_spans(text):
        if start <= position < end:
            return text[start:end], start, end
    return text, 0, len(text)


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019\-][A-Za-z0-9]+)*")


def _proximity(sentence: str, ent_start: int, ent_end: int, question_terms: set[str]) -> float:
    """How closely a candidate sits among the question's own words.

    Two CARDINALs in one sentence -- "Version 3.0 contains roughly one hundred
    and seventeen thousand synsets" -- score identically on evidence, type and
    word overlap. Proximity is what separates them: the real answer sits beside
    "synsets", the version number does not.

    Distance is averaged over every question term present rather than taken as
    a minimum, because a candidate wedged between two question words is a better
    answer than one merely touching a single common verb.
    """
    tokens = [
        (simple_stem(m.group().lower()), m.start(), m.end())
        for m in _WORD_RE.finditer(sentence)
    ]
    inside = [i for i, (_, s, e) in enumerate(tokens) if s < ent_end and e > ent_start]
    if not inside:
        return 0.0
    lo, hi = inside[0], inside[-1]

    distances = []
    for term in question_terms:
        hits = [
            i for i, (t, _, _) in enumerate(tokens)
            if t == term and not (lo <= i <= hi)
        ]
        if hits:
            distances.append(min(min(abs(i - lo), abs(i - hi)) for i in hits))
    if not distances:
        return 0.0
    return 1.0 / (1.0 + (sum(distances) / len(distances)) / 5.0)


def extract_answer(
    question: str,
    results: list[SearchResult],
    *,
    top_n: int = EXTRACT_FROM_TOP_N,
    min_confidence: float = MIN_ANSWER_CONFIDENCE,
    min_coverage: float = MIN_QUESTION_COVERAGE,
) -> Answer:
    """Pick the best answer span from the retrieved passages.

    Every candidate gets a score built from four parts, all in 0..1 and weighted
    to sum to 1.0, so the final number is directly readable as a confidence:

        evidence  how relevant the passage was          (from BM25)
        type      does the entity label fit the question
        context   does its sentence share question words, and does the
                  candidate sit close to them
        repeat    penalty when the candidate is a question word itself
    """
    qtype = classify_question(question)
    if not results:
        return Answer("", "none", 0.0, qtype.name)

    question_terms = set(analyze(question))
    best_retrieval = max(r.score for r in results) or 1.0

    # Coverage gate. Retrieval scores are relative, so the best of five
    # irrelevant passages still ranks first with a healthy-looking score. Asking
    # how much of the question a passage actually contains is an absolute test,
    # and it is what lets the system say "I don't know" instead of guessing.
    def covers(result: SearchResult) -> float:
        if not question_terms:
            return 0.0
        return len(question_terms & set(analyze(result.text))) / len(question_terms)

    considered = [r for r in results[:top_n] if covers(r) >= min_coverage]
    if not considered:
        return Answer("", "none", 0.0, qtype.name)

    scored: list[tuple[float, dict[str, float], Entity, SearchResult, tuple[str, int, int]]] = []

    if qtype.wants_span:
        for result in considered:
            evidence = result.score / best_retrieval
            for entity in find_entities(result.text):
                type_match = _type_match(entity.label, qtype)
                if type_match == 0.0:
                    continue

                sentence, s_start, s_end = _sentence_around(result.text, entity.start)
                sentence_terms = set(analyze(sentence))
                overlap = (
                    len(question_terms & sentence_terms) / len(question_terms)
                    if question_terms else 0.0
                )
                nearness = _proximity(
                    sentence,
                    entity.start - s_start,
                    entity.end - s_start,
                    question_terms,
                )
                # Context is half "does this sentence discuss the question" and
                # half "does the candidate sit among the question's words".
                context = 0.55 * overlap + 0.45 * nearness
                entity_terms = set(analyze(entity.text))
                repeat = 1.0 if entity_terms and entity_terms <= question_terms else 0.0

                parts = {
                    "evidence": round(W_EVIDENCE * evidence, 4),
                    "type": round(W_TYPE * type_match, 4),
                    "context": round(W_CONTEXT * context, 4),
                    "repeat": round(-W_REPEAT * repeat, 4),
                }
                scored.append((sum(parts.values()), parts, entity, result,
                               (sentence, s_start, s_end)))

    if scored:
        scored.sort(key=lambda item: -item[0])
        top_score, parts, entity, result, (sentence, _, _) = scored[0]

        if top_score >= min_confidence:
            # Distinct alternative surface forms, for the "other candidates" panel.
            alternatives: list[tuple[str, float]] = []
            for score, _, other, _, _ in scored[1:]:
                if other.text.lower() != entity.text.lower() and len(alternatives) < 3:
                    alternatives.append((other.text, round(max(score, 0.0), 3)))

            return Answer(
                text=entity.text,
                kind="span",
                confidence=round(min(max(top_score, 0.0), 1.0), 3),
                question_type=qtype.name,
                label=entity.label,
                sentence=sentence,
                doc_id=result.doc_id,
                doc_title=result.doc_title,
                doc_start=result.passage.start + entity.start,
                doc_end=result.passage.start + entity.end,
                passage_id=result.passage.passage_id,
                parts=parts,
                alternatives=alternatives,
            )

    # --- Fallback: quote the sentence that best covers the question ----------
    best_sentence = None
    for result in considered:
        evidence = result.score / best_retrieval
        for start, end in split_sentences_with_spans(result.text):
            sentence = result.text[start:end]
            overlap = (
                len(question_terms & set(analyze(sentence))) / len(question_terms)
                if question_terms else 0.0
            )
            score = W_EVIDENCE * evidence + (W_TYPE + W_CONTEXT) * overlap
            if best_sentence is None or score > best_sentence[0]:
                best_sentence = (score, sentence, start, end, result, overlap, evidence)

    if best_sentence is None:
        return Answer("", "none", 0.0, qtype.name)

    score, sentence, start, end, result, overlap, evidence = best_sentence
    return Answer(
        text=sentence.lstrip("# ").strip(),
        kind="sentence",
        confidence=round(min(max(score, 0.0), 1.0), 3),
        question_type=qtype.name,
        sentence=sentence,
        doc_id=result.doc_id,
        doc_title=result.doc_title,
        doc_start=result.passage.start + start,
        doc_end=result.passage.start + end,
        passage_id=result.passage.passage_id,
        parts={
            "evidence": round(W_EVIDENCE * evidence, 4),
            "context": round((W_TYPE + W_CONTEXT) * overlap, 4),
        },
    )
