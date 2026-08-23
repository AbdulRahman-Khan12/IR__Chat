"""Evaluation for IR__Chat.

A QA pipeline can fail in two separate places, so it has to be measured in two
separate places:

    retrieval    did the right document ever reach the extractor?
                 measured with recall@k and mean reciprocal rank

    extraction   given the right passage, was the right span chosen?
                 measured with exact match and token F1

Reporting only end-to-end accuracy hides which half is broken. A system with
95% recall and 40% exact match needs a better extractor; one with 50% recall
needs a better retriever. The numbers here tell you which.

A third measure matters for this system specifically: refusal accuracy. The
gold set contains questions the corpus cannot answer, and staying silent on
those is a correct answer, not a missing one.
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

from .config import EVAL_DIR
from .dialogue import DialogueManager

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalise_answer(text: str) -> str:
    """SQuAD's normalisation: lowercase, drop articles, punctuation, extra space.

    Without it "the Journal of Documentation" and "Journal of Documentation"
    count as different answers, which measures formatting rather than accuracy.
    """
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(predicted: str, gold: str) -> float:
    return float(normalise_answer(predicted) == normalise_answer(gold))


def token_f1(predicted: str, gold: str) -> float:
    """Token overlap F1, the standard partial-credit score for short answers.

    "Stephen Robertson" against "Stephen Robertson and Karen Sparck Jones"
    scores 0 on exact match but 0.57 here, which is a fairer description of
    what happened.
    """
    pred_tokens = normalise_answer(predicted).split()
    gold_tokens = normalise_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common: dict[str, int] = {}
    for token in pred_tokens:
        if token in gold_tokens:
            common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_gold(predicted: str, gold: str) -> float:
    """A looser check: is the gold answer inside what we said?

    Reported alongside exact match because a sentence-shaped answer to a
    definitional question is correct even though it can never match exactly.
    """
    return float(normalise_answer(gold) in normalise_answer(predicted))


@dataclass
class Result:
    question: str
    gold: str | None
    predicted: str
    route: str
    question_type: str
    em: float = 0.0
    f1: float = 0.0
    contains: float = 0.0
    gold_doc: str | None = None
    doc_rank: int | None = None      # rank of the first passage from the gold doc
    refused: bool = False

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.doc_rank if self.doc_rank else 0.0

    @property
    def correct(self) -> bool:
        if self.gold is None:
            return self.refused
        return bool(self.contains) or bool(self.em)


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)
    k: int = 5

    # --- slices --------------------------------------------------------------

    @property
    def answerable(self) -> list[Result]:
        return [r for r in self.results if r.gold is not None]

    @property
    def unanswerable(self) -> list[Result]:
        return [r for r in self.results if r.gold is None]

    def _mean(self, values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # --- headline numbers ----------------------------------------------------

    def summary(self) -> dict[str, float | int]:
        answerable = self.answerable
        ranked = [r for r in answerable if r.gold_doc]
        return {
            "questions": len(self.results),
            "answerable": len(answerable),
            "recall@k": self._mean([1.0 if r.doc_rank else 0.0 for r in ranked]),
            "MRR": self._mean([r.reciprocal_rank for r in ranked]),
            "exact_match": self._mean([r.em for r in answerable]),
            "token_f1": self._mean([r.f1 for r in answerable]),
            "answer_found": self._mean([r.contains for r in answerable]),
            "refusal_accuracy": self._mean(
                [1.0 if r.refused else 0.0 for r in self.unanswerable]
            ),
            "overall_correct": self._mean([1.0 if r.correct else 0.0 for r in self.results]),
        }

    def by_type(self) -> dict[str, dict[str, float]]:
        """Per-question-type breakdown, which is where the failures show up."""
        groups: dict[str, list[Result]] = {}
        for result in self.results:
            groups.setdefault(result.question_type, []).append(result)
        return {
            name: {
                "n": len(rows),
                "correct": self._mean([1.0 if r.correct else 0.0 for r in rows]),
                "f1": self._mean([r.f1 for r in rows if r.gold is not None]),
            }
            for name, rows in sorted(groups.items())
        }

    def by_route(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.route] = counts.get(result.route, 0) + 1
        return dict(sorted(counts.items()))

    def failures(self) -> list[Result]:
        return [r for r in self.results if not r.correct]


def load_pairs(path: Path | None = None) -> list[dict]:
    path = path or (EVAL_DIR / "qa_pairs.json")
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("pairs", [])


def evaluate(
    manager: DialogueManager,
    pairs: list[dict] | None = None,
    *,
    k: int = 5,
) -> Report:
    """Run every gold question through the full system.

    The dialogue state is reset before each question. Without that reset the
    topic slot from question 7 would silently answer question 8, and the score
    would measure the test order rather than the system.
    """
    pairs = pairs if pairs is not None else load_pairs()
    report = Report(k=k)

    for pair in pairs:
        manager.reset()
        question, gold = pair["q"], pair.get("a")
        reply = manager.respond(question)

        result = Result(
            question=question,
            gold=gold,
            predicted=reply.text if reply.has_evidence else "",
            route=reply.route,
            question_type=pair.get("type", "OTHER"),
            gold_doc=pair.get("doc"),
            refused=not reply.has_evidence,
        )

        if gold is not None:
            result.em = exact_match(result.predicted, gold)
            result.f1 = token_f1(result.predicted, gold)
            result.contains = contains_gold(result.predicted, gold)

            # Retrieval is scored independently of the answer, so a correct
            # retrieval still counts even when extraction picked badly.
            for rank, hit in enumerate(manager.retriever.search(question, top_k=k), start=1):
                if hit.doc_id == result.gold_doc:
                    result.doc_rank = rank
                    break

        report.results.append(result)

    manager.reset()
    return report
