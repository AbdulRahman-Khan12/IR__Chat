"""Passage construction for IR__Chat.

Retrieval never scores whole documents. A 300-word document about the
Transformer mentions "2017", "Google" and "attention" once each, and those
signals get averaged away against everything else in the file. A three-sentence
window keeps the signal concentrated, is short enough to show to the user as
evidence, and is short enough for Stage 3 to scan for answer spans.

Window 3 / stride 2 means consecutive passages overlap by one sentence. Overlap
matters because a question's evidence often straddles a sentence boundary: the
entity is named in one sentence and the date appears in the next.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import CHUNK_STRIDE_SENTENCES, CHUNK_WINDOW_SENTENCES, MIN_PASSAGE_CHARS
from .corpus import Corpus, Document
from .preprocess import split_sentences_with_spans


@dataclass(frozen=True)
class Passage:
    """A retrievable window of text plus everything needed to trace it back.

    `start` and `end` index into `Document.text`, which is why `Document` is
    frozen. Stage 3 uses them to locate an answer inside the source document;
    Stage 5 uses them to highlight it.
    """

    passage_id: str
    doc_id: str
    doc_title: str
    text: str
    start: int
    end: int
    index: int
    n_sentences: int

    @property
    def char_count(self) -> int:
        return len(self.text)

    def snippet(self, limit: int = 240) -> str:
        """Display form of the passage.

        Markdown heading markers are stripped here rather than in `text`. The
        heading words are genuine retrieval signal and must stay in the indexed
        text; the hashes are only noise on screen, and removing them from `text`
        would invalidate `start`/`end`.
        """
        flat = " ".join(word for word in self.text.split() if word.strip("#"))
        flat = flat.replace("# ", "")
        return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "\u2026"


def chunk_document(
    doc: Document,
    *,
    window: int = CHUNK_WINDOW_SENTENCES,
    stride: int = CHUNK_STRIDE_SENTENCES,
    min_chars: int = MIN_PASSAGE_CHARS,
) -> list[Passage]:
    """Slide a sentence window across one document."""
    if window < 1 or stride < 1:
        raise ValueError("window and stride must both be >= 1")

    spans = split_sentences_with_spans(doc.text)
    if not spans:
        return []

    passages: list[Passage] = []
    index = 0
    position = 0
    while position < len(spans):
        chunk = spans[position : position + window]
        start = chunk[0][0]
        end = chunk[-1][1]
        text = doc.text[start:end]

        # A short tail window is usually a heading or a stray line. Drop it,
        # unless it is all we have, in which case a short passage beats none.
        long_enough = len(text.strip()) >= min_chars
        if long_enough or (not passages and position + window >= len(spans)):
            passages.append(
                Passage(
                    passage_id=f"{doc.doc_id}::p{index:03d}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    text=text,
                    start=start,
                    end=end,
                    index=index,
                    n_sentences=len(chunk),
                )
            )
            index += 1

        if position + window >= len(spans):
            break
        position += stride

    return passages


def chunk_corpus(
    corpus: Corpus,
    *,
    window: int = CHUNK_WINDOW_SENTENCES,
    stride: int = CHUNK_STRIDE_SENTENCES,
    min_chars: int = MIN_PASSAGE_CHARS,
) -> list[Passage]:
    """Flatten the whole corpus into one passage list.

    Stage 2 indexes this list directly; passage position in the list is the
    document id used by the retriever.
    """
    passages: list[Passage] = []
    for doc in corpus:
        passages.extend(
            chunk_document(doc, window=window, stride=stride, min_chars=min_chars)
        )
    return passages


def verify_offsets(corpus: Corpus, passages: list[Passage]) -> list[str]:
    """Assert that every passage still slices cleanly out of its document.

    Cheap invariant, run in the Stage 1 self-check. If this ever fails, answer
    highlighting in Stage 5 would silently point at the wrong text.
    """
    problems: list[str] = []
    for p in passages:
        doc = corpus.get(p.doc_id)
        if doc is None:
            problems.append(f"{p.passage_id}: document {p.doc_id} is missing")
        elif doc.text[p.start : p.end] != p.text:
            problems.append(f"{p.passage_id}: offsets do not match the source text")
    return problems
