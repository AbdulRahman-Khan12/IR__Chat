"""Corpus layer for IR__Chat.

Owns three responsibilities:

* modelling a document (`Document`),
* holding a bounded collection of them (`Corpus`, hard capped at 25),
* turning a document back into bytes the user can download.

The cap is enforced in `Corpus.add` and nowhere else. Bundled loading and user
uploads both go through that method, so there is exactly one place where the
rule can be violated and exactly one place to check it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

from .config import (
    ALLOWED_EXTENSIONS,
    CORPUS_DIR,
    MAX_DOC_CHARS,
    MAX_DOCS,
    MIN_DOC_CHARS,
)
from .preprocess import normalise_text, tokenize


class CorpusError(Exception):
    """Base class for anything the corpus layer refuses to do."""


class CorpusLimitError(CorpusError):
    """Raised when adding a document would exceed MAX_DOCS."""


class DocumentRejected(CorpusError):
    """Raised when a single document is unusable (too short, duplicate, ...)."""


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def _slug(value: str) -> str:
    value = _SLUG_RE.sub("-", value.lower()).strip("-")
    return value or "doc"


def _derive_title(text: str, filename: str) -> str:
    """Prefer the document's own Markdown H1, then its first line, then the
    filename. Users recognise their documents by title, not by `doc_07`."""
    h1 = _H1_RE.search(text[:400])
    if h1:
        return h1.group(1).strip()
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:120]
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


@dataclass(frozen=True)
class Document:
    """An immutable unit of the searchable collection.

    Frozen on purpose: passages carry character offsets into `text`, so silently
    mutating the text later would invalidate every stored offset.
    """

    doc_id: str
    title: str
    text: str
    filename: str
    source: str = "bundled"          # "bundled" | "upload"
    path: Path | None = None
    checksum: str = ""

    @classmethod
    def from_text(
        cls,
        filename: str,
        raw_text: str,
        *,
        source: str = "bundled",
        path: Path | None = None,
        doc_id: str | None = None,
    ) -> "Document":
        text = normalise_text(raw_text)
        if len(text) < MIN_DOC_CHARS:
            raise DocumentRejected(
                f"{filename}: only {len(text)} characters, minimum is {MIN_DOC_CHARS}."
            )
        if len(text) > MAX_DOC_CHARS:
            raise DocumentRejected(
                f"{filename}: {len(text):,} characters, maximum is {MAX_DOC_CHARS:,}."
            )
        checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return cls(
            doc_id=doc_id or _slug(Path(filename).stem),
            title=_derive_title(text, filename),
            text=text,
            filename=filename,
            source=source,
            path=path,
            checksum=checksum,
        )

    @classmethod
    def from_path(cls, path: Path, *, source: str = "bundled") -> "Document":
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise DocumentRejected(
                f"{path.name}: {path.suffix or 'no extension'} is not supported "
                f"({', '.join(sorted(ALLOWED_EXTENSIONS))})."
            )
        return cls.from_text(
            path.name, path.read_text(encoding="utf-8", errors="replace"),
            source=source, path=path,
        )

    # --- derived properties --------------------------------------------------

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(tokenize(self.text))

    def preview(self, limit: int = 180) -> str:
        body = self.text
        h1 = _H1_RE.match(body)
        if h1:
            body = body[h1.end():].lstrip()
        body = " ".join(body.split())
        return body if len(body) <= limit else body[: limit - 1].rstrip() + "\u2026"

    # --- download ------------------------------------------------------------

    def download_payload(self) -> bytes:
        """Bytes for the download button.

        Reads the file from disk when there is one so the user gets back exactly
        what was shipped; falls back to the normalised text for uploads held
        only in session memory.
        """
        if self.path is not None and self.path.exists():
            return self.path.read_bytes()
        return self.text.encode("utf-8")

    @property
    def download_name(self) -> str:
        return self.filename or f"{self.doc_id}.txt"


class Corpus:
    """A bounded, de-duplicated, ordered set of documents."""

    def __init__(self, max_docs: int = MAX_DOCS) -> None:
        self.max_docs = max_docs
        self._docs: dict[str, Document] = {}
        self._checksums: dict[str, str] = {}   # checksum -> doc_id

    # --- container protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self._docs)

    def __iter__(self) -> Iterator[Document]:
        return iter(self._docs.values())

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._docs

    @property
    def documents(self) -> tuple[Document, ...]:
        return tuple(self._docs.values())

    @property
    def remaining_slots(self) -> int:
        return max(0, self.max_docs - len(self._docs))

    @property
    def is_full(self) -> bool:
        return len(self._docs) >= self.max_docs

    def get(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    # --- mutation ------------------------------------------------------------

    def add(self, doc: Document) -> Document:
        """The one and only entry point for growing the corpus.

        Enforces the 25-document cap, rejects byte-identical duplicates, and
        makes `doc_id` unique so two files named `notes.txt` can coexist.
        """
        if self.is_full:
            raise CorpusLimitError(
                f"The corpus already holds {self.max_docs} documents. "
                f"Remove one before adding \u201c{doc.title}\u201d."
            )
        if doc.checksum in self._checksums:
            existing = self._docs[self._checksums[doc.checksum]]
            raise DocumentRejected(
                f"\u201c{doc.title}\u201d is identical to \u201c{existing.title}\u201d, "
                "which is already indexed."
            )

        doc_id = doc.doc_id
        if doc_id in self._docs:
            suffix = 2
            while f"{doc_id}-{suffix}" in self._docs:
                suffix += 1
            doc_id = f"{doc_id}-{suffix}"
            doc = replace(doc, doc_id=doc_id)

        self._docs[doc_id] = doc
        self._checksums[doc.checksum] = doc_id
        return doc

    def remove(self, doc_id: str) -> bool:
        doc = self._docs.pop(doc_id, None)
        if doc is None:
            return False
        self._checksums.pop(doc.checksum, None)
        return True

    def clear(self) -> None:
        self._docs.clear()
        self._checksums.clear()

    # --- reporting -----------------------------------------------------------

    @property
    def fingerprint(self) -> str:
        """Stable hash of the whole collection.

        Stage 2 caches the index against this: change the corpus and the index
        rebuilds, leave it alone and it does not.
        """
        joined = "|".join(sorted(d.checksum for d in self._docs.values()))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    def stats(self) -> dict[str, int | str]:
        words = [d.word_count for d in self._docs.values()]
        return {
            "documents": len(self._docs),
            "slots_free": self.remaining_slots,
            "total_words": sum(words),
            "median_words": sorted(words)[len(words) // 2] if words else 0,
            "uploaded": sum(1 for d in self._docs.values() if d.source == "upload"),
            "fingerprint": self.fingerprint,
        }

    def to_rows(self) -> list[dict[str, object]]:
        """Flat rows for the Stage 5 document table."""
        return [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "words": d.word_count,
                "source": d.source,
                "file": d.download_name,
            }
            for d in self._docs.values()
        ]


def load_bundled_corpus(
    directory: Path = CORPUS_DIR, *, max_docs: int = MAX_DOCS
) -> tuple[Corpus, list[str]]:
    """Load the shipped corpus from disk.

    Returns the corpus plus a list of human-readable notes about anything that
    was skipped. Nothing raises: a single bad file must never stop the app from
    starting.
    """
    corpus = Corpus(max_docs=max_docs)
    notes: list[str] = []

    if not directory.exists():
        return corpus, [f"Corpus folder not found: {directory}"]

    paths = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )
    for path in paths:
        try:
            corpus.add(Document.from_path(path))
        except CorpusLimitError as exc:
            notes.append(str(exc))
            break
        except CorpusError as exc:
            notes.append(str(exc))
    return corpus, notes
