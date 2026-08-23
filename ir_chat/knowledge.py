"""Knowledge-based question answering for IR__Chat.

The other half of the QA story. Stages 2 and 3 read text and guess a span.
This module does not read anything: it turns the question into a structured
lookup and returns a stored fact.

    "Who wrote ELIZA?"  ->  subject = ELIZA, relation = created_by
                        ->  facts[(ELIZA, created_by)] = "Joseph Weizenbaum"

The trade-off is the classic one, and it is worth stating in the report:

    knowledge-based   exact, always citable, never hallucinates
                      but can only answer what somebody entered

    IR-based          answers anything the corpus mentions
                      but can pick the wrong span

IR__Chat uses both. The dialogue manager tries the knowledge base first because
a lookup that succeeds is always right, then falls back to retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import KB_DIR


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    obj: str
    doc_id: str

    def as_triple(self) -> str:
        return f"({self.subject}) --[{self.relation}]--> ({self.obj})"


@dataclass(frozen=True)
class KBAnswer:
    """A fact retrieved by lookup, plus the parse that found it."""

    fact: Fact
    subject: str
    relation: str
    relation_label: str
    matched_cue: str

    @property
    def text(self) -> str:
        return self.fact.obj

    @property
    def doc_id(self) -> str:
        return self.fact.doc_id

    def explain(self) -> list[str]:
        return [
            f"subject recognised    {self.subject}",
            f"relation recognised   {self.relation}  (cue: \u201c{self.matched_cue}\u201d)",
            f"fact looked up        {self.fact.as_triple()}",
            f"source document       {self.fact.doc_id}",
        ]


class KnowledgeBase:
    """Loads triples from JSON and answers (subject, relation) lookups."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (KB_DIR / "facts.json")
        self.facts: list[Fact] = []
        self.relations: dict[str, dict] = {}
        self.aliases: dict[str, str] = {}
        self._index: dict[tuple[str, str], Fact] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.relations = data.get("relations", {})
        self.aliases = {k.lower(): v for k, v in data.get("aliases", {}).items()}
        for row in data.get("facts", []):
            fact = Fact(row["s"], row["r"], row["o"], row.get("doc", ""))
            self.facts.append(fact)
            self._index[(fact.subject.lower(), fact.relation)] = fact

    # --- introspection -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.facts)

    @property
    def subjects(self) -> list[str]:
        seen: dict[str, None] = {}
        for fact in self.facts:
            seen.setdefault(fact.subject, None)
        return list(seen)

    def facts_about(self, subject: str) -> list[Fact]:
        return [f for f in self.facts if f.subject.lower() == subject.lower()]

    def stats(self) -> dict[str, int]:
        return {
            "facts": len(self.facts),
            "subjects": len(self.subjects),
            "relations": len(self.relations),
            "documents_covered": len({f.doc_id for f in self.facts}),
        }

    # --- parsing -------------------------------------------------------------

    def find_subject(self, question: str) -> str | None:
        """Longest subject name appearing in the question.

        Longest-first matters: "Google Knowledge Graph" must win over "Google",
        or every question about the graph would be answered about the company.
        """
        text = question.lower()
        for alias, canonical in sorted(self.aliases.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{re.escape(alias)}\b", text):
                text = text.replace(alias, canonical.lower())
                break

        best: str | None = None
        for subject in self.subjects:
            if re.search(rf"\b{re.escape(subject.lower())}\b", text):
                if best is None or len(subject) > len(best):
                    best = subject
        return best

    def find_relation(self, question: str) -> tuple[str, str] | None:
        """Match the question against relation cue phrases.

        Longest cue wins, so "who wrote" beats a bare "who" and "when was"
        beats "when".
        """
        text = question.lower()
        best: tuple[str, str] | None = None
        for name, spec in self.relations.items():
            for cue in spec.get("cues", []):
                if cue in text and (best is None or len(cue) > len(best[1])):
                    best = (name, cue)
        return best

    def answer(self, question: str) -> KBAnswer | None:
        """Full lookup: question -> (subject, relation) -> fact, or nothing.

        Returning None is the normal case, not an error. The knowledge base
        covers a few dozen facts; everything else is retrieval's job.
        """
        subject = self.find_subject(question)
        if subject is None:
            return None
        relation = self.find_relation(question)
        if relation is None:
            return None

        name, cue = relation
        fact = self._index.get((subject.lower(), name))
        if fact is None:
            return None

        return KBAnswer(
            fact=fact,
            subject=subject,
            relation=name,
            relation_label=self.relations.get(name, {}).get("label", name),
            matched_cue=cue,
        )
