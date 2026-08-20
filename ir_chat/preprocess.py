"""Text preprocessing for IR__Chat.

Deliberately dependency-free. Two reasons:

1. Streamlit Community Cloud boots a fresh container on every deploy. Anything
   that needs `nltk.download()` at import time is a startup failure waiting to
   happen. NLTK and spaCy enter the project in Stage 3, where their linguistic
   models actually earn their weight (NER for answer extraction).
2. Every step here is one you can read and reason about, which matters for the
   "Pipeline Design & Methodology" criterion.

The important design choice in this module is that `split_sentences_with_spans`
returns *character offsets*, not just strings. Those offsets survive all the way
to Stage 3, where an extracted answer has to be highlighted inside the original
document text.
"""

from __future__ import annotations

import re
import unicodedata

# --- Normalisation -----------------------------------------------------------

_WS_RE = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalise_text(text: str) -> str:
    """Canonicalise unicode and whitespace without changing the meaning.

    Applied once, at load time, so that character offsets computed later are
    offsets into a stable string.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# --- Sentence segmentation ---------------------------------------------------

# Tokens that end in a period but do not end a sentence.
_ABBREVIATIONS = frozenset(
    """
    dr mr mrs ms prof sr jr st vs etc al fig no inc ltd co corp dept univ
    e.g i.e approx est ca cf viz u.s u.k a.m p.m ph.d b.sc m.sc
    jan feb mar apr jun jul aug sep sept oct nov dec
    """.split()
)

# A candidate boundary: sentence punctuation, optional closing quote/bracket,
# then whitespace.
_BOUNDARY_RE = re.compile(r"[.!?]+[\"')\]]*\s+")
_WORD_BEFORE_RE = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def _is_real_boundary(text: str, match: re.Match) -> bool:
    """Reject the three classic false positives: abbreviations, initials, and a
    following lowercase word (which almost always means the period was not a
    full stop)."""
    before = text[: match.start()]
    word_match = _WORD_BEFORE_RE.search(before)
    if word_match:
        word = word_match.group(1)
        if word.lower().rstrip(".") in _ABBREVIATIONS:
            return False
        if len(word) == 1 and word.isupper():  # middle initial, e.g. "George A. Miller"
            return False

    after = text[match.end() : match.end() + 1]
    if after and after.islower():
        return False
    return True


def _split_block(block: str, offset: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(block):
        if not _is_real_boundary(block, match):
            continue
        end = match.start() + len(match.group().rstrip())
        spans.append((start, end))
        start = match.end()

    tail = block[start:]
    if tail.strip():
        spans.append((start, len(block.rstrip())))

    cleaned: list[tuple[int, int]] = []
    for s, e in spans:
        while s < e and block[s].isspace():
            s += 1
        while e > s and block[e - 1].isspace():
            e -= 1
        if e > s:
            cleaned.append((s + offset, e + offset))
    return cleaned


def split_sentences_with_spans(text: str) -> list[tuple[int, int]]:
    """Return `(start, end)` character offsets of every sentence in `text`.

    Blank lines are treated as hard boundaries first. That handles Markdown
    headings, which end without punctuation and would otherwise be glued to the
    paragraph beneath them.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for para in _PARAGRAPH_RE.split(text):
        block_start = text.find(para, cursor)
        if block_start == -1:  # defensive; should not happen
            block_start = cursor
        spans.extend(_split_block(para, block_start))
        cursor = block_start + len(para)
    return spans


def split_sentences(text: str) -> list[str]:
    """Convenience wrapper when the offsets are not needed."""
    return [text[s:e] for s, e in split_sentences_with_spans(text)]


# --- Tokenisation ------------------------------------------------------------

# Keeps internal apostrophes and hyphens ("state-of-the-art", "Turing's") and
# keeps digits, because factoid answers are very often years or counts.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019\-][A-Za-z0-9]+)*")


def tokenize(text: str, *, lower: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text)
    return [t.lower() for t in tokens] if lower else tokens


STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he her here hers
    herself him himself his how i if in into is isn't it its itself just me
    more most mustn't my myself no nor not of off on once only or other ought
    our ours ourselves out over own same shan't she should shouldn't so some
    such than that the their theirs them themselves then there these they this
    those through to too under until up very was wasn't we were weren't what
    when where which while who whom why with won't would wouldn't you your
    yours yourself yourselves
    """.split()
)

# Question words are stopwords for matching, but Stage 3 needs them to decide
# the expected answer type, so they are kept in a separate set and never
# silently dropped from the raw question.
QUESTION_WORDS = frozenset(
    "who whom whose what which when where why how many much long".split()
)

_PLURAL_RULES = (("ies", "y"), ("sses", "ss"), ("shes", "sh"), ("ches", "ch"), ("xes", "x"))
_VERB_SUFFIXES = ("ingly", "edly", "ing", "ed", "ly")
_NO_STRIP_S = ("ss", "us", "is", "as", "os")


def simple_stem(token: str) -> str:
    """A conservative suffix stripper.

    It is not Porter. It does not need to be: the only job is to make
    "retrieve/retrieved/retrieving" collide on one index key. Over-stemming
    costs precision, so the rules stay shallow and every rule keeps a stem of at
    least three characters.
    """
    if len(token) <= 3 or any(c.isdigit() for c in token):
        return token

    for suffix, replacement in _PLURAL_RULES:
        if token.endswith(suffix) and len(token) - len(suffix) + len(replacement) >= 3:
            return token[: -len(suffix)] + replacement

    for suffix in _VERB_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]

    if token.endswith("s") and not token.endswith(_NO_STRIP_S):
        return token[:-1]

    return token


def analyze(
    text: str,
    *,
    remove_stopwords: bool = True,
    stem: bool = True,
    keep_question_words: bool = False,
) -> list[str]:
    """The full term pipeline: tokenize -> lowercase -> stopwords -> stem.

    This single function is what makes the index and the query comparable. Both
    sides must go through it, otherwise a query term will never match its own
    document term.
    """
    terms = tokenize(text)
    if remove_stopwords:
        allowed = QUESTION_WORDS if keep_question_words else frozenset()
        terms = [t for t in terms if t not in STOPWORDS or t in allowed]
    if stem:
        terms = [simple_stem(t) for t in terms]
    return [t for t in terms if t]
