"""Retrieval for IR__Chat.

One inverted index, two scoring formulas over the same statistics.

The index stores, for every term, the list of passages containing it. A query
therefore touches only the passages that share a word with it, never all 60.
That is the entire reason inverted indexes exist, and it is the difference
between a search that scales and one that does not.

BM25 is the default and does the real work. TF-IDF cosine is included because
it costs about fifteen extra lines over the same counts and makes the
comparison in Stage 5 concrete rather than theoretical.

--- The BM25 formula --------------------------------------------------------

    score(q, d) = SUM over query terms t of

                                        f(t,d) * (k1 + 1)
              IDF(t)  *  ------------------------------------------------
                          f(t,d) + k1 * (1 - b + b * len(d) / avg_len)

    IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

Read it as three independent ideas multiplied together:

* IDF        a rare term is worth more than a common one
* saturation seeing a term ten times is not ten times better than once (k1)
* length     a long passage should not win by sheer size (b)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .chunker import Passage, chunk_corpus
from .config import BM25_B, BM25_K1, DEFAULT_TOP_K
from .corpus import Corpus
from .preprocess import analyze


@dataclass(frozen=True)
class SearchResult:
    """One ranked passage plus the reason it ranked there.

    `term_scores` is what makes the system explainable: it holds each query
    term's individual contribution to the total, so the interface can say
    "this ranked first because of bm25 (4.1) and robertson (3.2)" instead of
    showing a bare number the user has to trust.
    """

    passage: Passage
    score: float
    rank: int
    term_scores: dict[str, float]

    @property
    def doc_id(self) -> str:
        return self.passage.doc_id

    @property
    def doc_title(self) -> str:
        return self.passage.doc_title

    @property
    def text(self) -> str:
        return self.passage.text

    def snippet(self, limit: int = 240) -> str:
        return self.passage.snippet(limit)

    @property
    def matched_terms(self) -> list[str]:
        """Query terms this passage actually contains, strongest first."""
        return [t for t, _ in sorted(self.term_scores.items(), key=lambda kv: -kv[1])]


class Retriever:
    """Builds the index once, then answers queries against it."""

    def __init__(
        self,
        corpus: Corpus,
        *,
        passages: list[Passage] | None = None,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        self.corpus = corpus
        self.passages = passages if passages is not None else chunk_corpus(corpus)
        self.k1 = k1
        self.b = b
        # Carried from the corpus so Stage 5 can cache the index and rebuild it
        # only when the document set actually changes.
        self.fingerprint = corpus.fingerprint
        self._build()

    # --- indexing ------------------------------------------------------------

    def _build(self) -> None:
        self._tf: list[Counter[str]] = []          # per passage: term -> count
        self._length: list[int] = []               # per passage: total terms
        self._postings: dict[str, list[int]] = defaultdict(list)

        for i, passage in enumerate(self.passages):
            counts = Counter(analyze(passage.text))
            self._tf.append(counts)
            self._length.append(sum(counts.values()))
            for term in counts:
                self._postings[term].append(i)

        self.n = len(self.passages)
        self.avg_length = (sum(self._length) / self.n) if self.n else 0.0

        # Two IDF variants, because the two scorers define it differently.
        # BM25's version stays non-negative even for terms in every passage.
        self._idf: dict[str, float] = {}
        self._idf_tfidf: dict[str, float] = {}
        for term, postings in self._postings.items():
            df = len(postings)
            self._idf[term] = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            self._idf_tfidf[term] = math.log(self.n / df)

        # Precomputed vector lengths for TF-IDF cosine. Doing this once at build
        # time keeps query-time cost proportional to the query, not the corpus.
        self._norm: list[float] = []
        for counts in self._tf:
            total = sum(
                ((1 + math.log(tf)) * self._idf_tfidf[t]) ** 2
                for t, tf in counts.items()
            )
            self._norm.append(math.sqrt(total) or 1.0)

    # --- query handling ------------------------------------------------------

    def analyze_query(self, query: str) -> tuple[list[str], list[str]]:
        """Split a query into terms the index knows and terms it has never seen.

        The unknown list is not waste. It is the honest explanation for an empty
        result: "no document mentions 'quantum'" is far more useful to a user
        than a blank screen.
        """
        terms = analyze(query)
        known = [t for t in terms if t in self._postings]
        unknown = [t for t in terms if t not in self._postings]
        return known, unknown

    def _score_bm25(self, term: str, passage_idx: int) -> float:
        tf = self._tf[passage_idx][term]
        length_ratio = self._length[passage_idx] / self.avg_length
        denominator = tf + self.k1 * (1 - self.b + self.b * length_ratio)
        return self._idf[term] * (tf * (self.k1 + 1)) / denominator

    def _score_tfidf(self, term: str, passage_idx: int) -> float:
        tf = self._tf[passage_idx][term]
        weight = (1 + math.log(tf)) * self._idf_tfidf[term]
        return weight * self._idf_tfidf[term] / self._norm[passage_idx]

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        scorer: str = "bm25",
    ) -> list[SearchResult]:
        """Rank passages against the query.

        Only passages appearing in a query term's postings list are ever scored,
        so cost grows with the number of query terms rather than corpus size.
        """
        if scorer not in {"bm25", "tfidf"}:
            raise ValueError("scorer must be 'bm25' or 'tfidf'")

        known, _ = self.analyze_query(query)
        if not known:
            return []

        score_term = self._score_bm25 if scorer == "bm25" else self._score_tfidf
        contributions: dict[int, dict[str, float]] = defaultdict(dict)
        for term in set(known):
            for passage_idx in self._postings[term]:
                contributions[passage_idx][term] = score_term(term, passage_idx)

        ranked = sorted(
            contributions.items(),
            key=lambda item: (-sum(item[1].values()), item[0]),
        )[:top_k]

        return [
            SearchResult(
                passage=self.passages[idx],
                score=sum(terms.values()),
                rank=rank,
                term_scores=terms,
            )
            for rank, (idx, terms) in enumerate(ranked, start=1)
        ]

    def search_documents(
        self, query: str, *, top_k: int = 3, scorer: str = "bm25"
    ) -> list[SearchResult]:
        """Rank documents by their single best passage.

        Useful for "which document covers this?", where five passages from the
        same file would just be noise.
        """
        results = self.search(query, top_k=self.n, scorer=scorer)
        best: dict[str, SearchResult] = {}
        for result in results:
            if result.doc_id not in best:
                best[result.doc_id] = result
        top = list(best.values())[:top_k]
        return [
            SearchResult(r.passage, r.score, rank, r.term_scores)
            for rank, r in enumerate(top, start=1)
        ]

    # --- explanation ---------------------------------------------------------

    def explain(self, result: SearchResult) -> list[dict[str, float | str | int]]:
        """Break one result into a per-term table: tf, df, idf, contribution.

        This is the evidence behind the ranking, and the thing to point at when
        asked why the system chose a passage.
        """
        idx = self.passages.index(result.passage)
        rows = []
        for term, contribution in sorted(result.term_scores.items(), key=lambda kv: -kv[1]):
            rows.append(
                {
                    "term": term,
                    "tf": self._tf[idx][term],
                    "df": len(self._postings[term]),
                    "idf": round(self._idf[term], 3),
                    "score": round(contribution, 3),
                }
            )
        return rows

    def stats(self) -> dict[str, int | float]:
        return {
            "passages": self.n,
            "vocabulary": len(self._postings),
            "avg_passage_terms": round(self.avg_length, 1),
            "k1": self.k1,
            "b": self.b,
        }


def build_retriever(corpus: Corpus, **kwargs) -> Retriever:
    """Convenience constructor, kept so callers never import chunker directly."""
    return Retriever(corpus, **kwargs)
