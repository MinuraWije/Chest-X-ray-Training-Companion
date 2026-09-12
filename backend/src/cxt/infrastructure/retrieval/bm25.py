"""Okapi BM25, pure Python, zero new runtime dependencies.

The corpus is 50-80 short chunks. A dense retriever would mean a
sentence-transformer, torch on the API host, and a model download in the
clone-and-run path — all to rank fewer documents than fit on a screen. BM25
is the honest choice at this scale, and it has the property that matters for
a thesis artefact: every number it produces can be derived by hand.

Two decisions worth stating, because both are places this quietly goes wrong.

**Length normalisation.** `b=0.75` scales a document's score by how its length
compares to the corpus average. Drop the average, or divide by the wrong
length, and the ranker drifts toward returning whichever chunk is longest. It
still returns fluent medical prose, so nothing downstream looks broken and
every retrieval figure in the report is meaningless. `tests/.../test_bm25.py`
pins the expected ordering against a toy corpus for exactly this reason.

**IDF.** The textbook Robertson-Sparck-Jones form goes negative for a term
appearing in more than half the corpus — with 60 chunks about six findings,
"lung" and "chest" will. A negative contribution means a document is punished
for containing a query term, which is not a behaviour anyone intends. The
`+1` smoothed variant used here is always positive and is what Lucene ships.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Final

__all__ = ["BM25Index", "normalise_score", "tokenize", "K1", "B", "SCORE_SATURATION_K"]

#: Term-frequency saturation. Standard range is 1.2-2.0; 1.2 is the usual
#: default and is right for short chunks, where a term appearing five times
#: rather than twice says more about the chunk's length than its aboutness.
K1: Final[float] = 1.2

#: Length normalisation. 0 disables it, 1 applies it fully; 0.75 is standard.
B: Final[float] = 0.75

#: The `k` in `score = bm25 / (bm25 + k)`. Chosen so a solidly-matching chunk
#: lands near the middle of the range rather than pinned at either end: with
#: k=4, a raw BM25 of 4 reads as 0.5 and a strong 12 as 0.75. It is the only
#: free parameter in the displayed score, and it is a monotone reparametrisation
#: — it can never change the *ordering*, only where the abstention floor sits.
SCORE_SATURATION_K: Final[float] = 4.0

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on anything that is not a letter or digit.

    No stemming and no stop-word list. Both are tunable knobs that would need
    defending in the write-up, and at 60 chunks neither buys enough to be worth
    the paragraph. Digits are kept because "0.50" — the cardiothoracic
    threshold — is a term the corpus genuinely uses.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def normalise_score(raw: float) -> float:
    """Map a raw BM25 score into [0, 1) — saturating and absolute.

    Deliberately *not* normalised against the top hit. Top-1 normalisation
    makes the best result 1.0 whatever its raw score, so the UI would render
    "100% match" next to a chunk BM25 barely liked, in a tool a medical
    student is meant to trust. It also destroys thresholding: if the top hit
    is always 1.0, there is no floor below which the system can abstain.
    """
    if math.isnan(raw) or raw <= 0.0:
        # NaN and a negative IDF contribution both mean "no evidence here",
        # never "best match". `RetrievedPassage` would reject either anyway.
        return 0.0
    if math.isinf(raw):
        return 1.0
    return raw / (raw + SCORE_SATURATION_K)


@dataclass(frozen=True, slots=True)
class BM25Index:
    """An immutable inverted index over `{doc_id: text}`.

    Built once at process start and shared; `search` does not mutate it.
    """

    doc_lengths: dict[str, int]
    term_frequencies: dict[str, Counter[str]]
    document_frequencies: dict[str, int]
    average_length: float
    _doc_order: tuple[str, ...] = field(default=())

    @property
    def size(self) -> int:
        return len(self.doc_lengths)

    @classmethod
    def build(cls, documents: dict[str, str]) -> BM25Index:
        term_frequencies: dict[str, Counter[str]] = {}
        doc_lengths: dict[str, int] = {}
        document_frequencies: Counter[str] = Counter()
        for doc_id, text in documents.items():
            tokens = tokenize(text)
            if not tokens:
                # A chunk that tokenizes to nothing can never match, and keeping
                # it would drag the average length down for everything else.
                continue
            counts = Counter(tokens)
            term_frequencies[doc_id] = counts
            doc_lengths[doc_id] = len(tokens)
            document_frequencies.update(counts.keys())
        total = sum(doc_lengths.values())
        return cls(
            doc_lengths=doc_lengths,
            term_frequencies=term_frequencies,
            document_frequencies=dict(document_frequencies),
            average_length=(total / len(doc_lengths)) if doc_lengths else 0.0,
            _doc_order=tuple(term_frequencies),
        )

    def _idf(self, term: str) -> float:
        """Smoothed inverse document frequency — always positive.

        `log(1 + (N - df + 0.5) / (df + 0.5))`. The unsmoothed form drops below
        zero once a term is in more than half the corpus, which would subtract
        score from a document for containing a query term.
        """
        n = len(self.doc_lengths)
        df = self.document_frequencies.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, query_terms: list[str], doc_id: str) -> float:
        counts = self.term_frequencies.get(doc_id)
        if not counts:
            return 0.0
        length = self.doc_lengths[doc_id]
        # The length ratio. When `average_length` is 0 the corpus is empty and
        # we never reach here.
        norm = 1.0 - B + B * (length / self.average_length)
        total = 0.0
        for term in set(query_terms):
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            total += self._idf(term) * (tf * (K1 + 1.0)) / (tf + K1 * norm)
        return total

    def search(self, query: str, *, limit: int | None = None) -> list[tuple[str, float]]:
        """Rank documents by raw BM25, descending. Zero-scoring docs are dropped.

        Ties break on the corpus insertion order rather than arbitrarily, so
        two runs of one query give one answer — otherwise the eval would be
        measuring dict iteration and reporting it as retrieval quality.
        """
        terms = tokenize(query)
        if not terms or not self.doc_lengths:
            return []
        scored: list[tuple[str, float, int]] = []
        for position, doc_id in enumerate(self._doc_order):
            value = self.score(terms, doc_id)
            if value > 0.0:
                scored.append((doc_id, value, position))
        scored.sort(key=lambda row: (-row[1], row[2]))
        trimmed = scored if limit is None else scored[:limit]
        return [(doc_id, value) for doc_id, value, _ in trimmed]
