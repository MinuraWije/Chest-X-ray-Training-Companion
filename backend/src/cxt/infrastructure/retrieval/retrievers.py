"""The Phase 2 retrieval arms, behind one port.

Four implementations, one per experimental arm, all satisfying `IRetriever`:

- `NullRetriever`      — returns nothing. The p2.4 and p3.0-empty arms.
- `CategoryRetriever`  — the case's confirmed category to its chunks. The
                         **oracle**: an upper bound, not a retrieval result.
- `BM25Retriever`      — unfiltered BM25 over the student's free text. The
                         only arm whose relevance number is non-circular.
- `MisconceptionRetriever` — BM25 plus a boost on what the student got wrong.

Why the oracle's precision is not reported: it filters on `findings`, which is
the same tag precision would be scored against, so precision@k is 1.0 by
construction and falls below 1.0 only when the corpus holds fewer than k chunks
for that category — which measures corpus coverage while calling it retrieval
quality. The oracle answers a different and still useful question: does
supplying *correct* evidence improve the feedback at all?

All four abstain below an absolute score floor rather than padding the result
with weak matches. Showing a student a barely-relevant passage "for
completeness" in a medical teaching tool teaches them to discount citations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

from cxt.domain.entities.feedback import RetrievedPassage
from cxt.domain.ports.retriever import RetrievalQuery
from cxt.infrastructure.retrieval.bm25 import BM25Index, normalise_score
from cxt.infrastructure.retrieval.corpus import Corpus, CorpusChunk

logger = logging.getLogger(__name__)

__all__ = [
    "NullRetriever",
    "CategoryRetriever",
    "BM25Retriever",
    "MisconceptionRetriever",
    "DEFAULT_SCORE_FLOOR",
    "MISCONCEPTION_BOOST",
]

#: Normalised score below which a passage is not worth showing. 0.25
#: corresponds to a raw BM25 of about 1.3 — a single mid-frequency term match,
#: which in practice is a chunk that shares a word with the query and nothing
#: else. Absolute, which is only possible because `normalise_score` is
#: saturating rather than top-1 normalised.
DEFAULT_SCORE_FLOOR: Final[float] = 0.25

#: Multiplier applied to a chunk whose `confusions` tag names something the
#: student got wrong. 1.5 is enough to reorder within a band of similar scores
#: without letting a boosted irrelevant chunk overtake a strong lexical match —
#: the boost is meant to break ties pedagogically, not to override BM25.
MISCONCEPTION_BOOST: Final[float] = 1.5


def _to_passage(chunk: CorpusChunk, score: float) -> RetrievedPassage:
    return RetrievedPassage(
        text=chunk.text,
        source=chunk.source if not chunk.title else f"{chunk.source} — {chunk.title}",
        score=score,
        id=chunk.id,
        url=chunk.url,
    )


@dataclass(frozen=True, slots=True)
class NullRetriever:
    """Returns no passages. The Phase 1 behaviour and the p3.0-empty control."""

    def retrieve(
        self, *, query: RetrievalQuery, top_k: int = 3
    ) -> tuple[RetrievedPassage, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class CategoryRetriever:
    """Oracle arm: confirmed category -> its chunks, ranked by BM25 within.

    Uses the answer key, so it is an upper bound on what retrieval could
    contribute, not a measurement of retrieval. Reported as such.
    """

    corpus: Corpus

    def retrieve(
        self, *, query: RetrievalQuery, top_k: int = 3
    ) -> tuple[RetrievedPassage, ...]:
        if not query.confirmed_categories:
            return ()
        candidates = [
            chunk
            for chunk in self.corpus.chunks
            if any(c in chunk.findings for c in query.confirmed_categories)
        ]
        if not candidates:
            return ()
        # Rank within the filtered set by the student's words, so which chunk
        # the student sees still depends on what they wrote. Falling back to
        # corpus order when their text matches nothing keeps the arm total.
        index = BM25Index.build({c.id: c.search_text for c in candidates})
        ranked = index.search(query.free_text, limit=top_k)
        by_id = {c.id: c for c in candidates}
        if ranked:
            return tuple(
                _to_passage(by_id[doc_id], normalise_score(raw)) for doc_id, raw in ranked
            )
        return tuple(_to_passage(c, 0.0) for c in candidates[:top_k])


@dataclass(frozen=True, slots=True)
class BM25Retriever:
    """Unfiltered BM25 over the whole corpus, keyed on the student's free text.

    The only arm that yields a relevance figure which is not circular: nothing
    about the case's answer key participates in the ranking, so tag overlap
    between the retrieved chunk and the confirmed finding is a real measurement.
    """

    corpus: Corpus
    score_floor: float = DEFAULT_SCORE_FLOOR
    _index: BM25Index = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        index = BM25Index.build({c.id: c.search_text for c in self.corpus.chunks})
        object.__setattr__(self, "_index", index)

    def _rank(self, query: RetrievalQuery) -> list[tuple[str, float]]:
        return self._index.search(query.free_text)

    def retrieve(
        self, *, query: RetrievalQuery, top_k: int = 3
    ) -> tuple[RetrievedPassage, ...]:
        by_id = self.corpus.by_id
        passages: list[RetrievedPassage] = []
        for doc_id, raw in self._rank(query):
            score = normalise_score(raw)
            if score < self.score_floor:
                # Ranked descending, so the first failure ends the list.
                break
            chunk = by_id.get(doc_id)
            if chunk is not None:
                passages.append(_to_passage(chunk, score))
            if len(passages) >= top_k:
                break
        return tuple(passages)


@dataclass(frozen=True, slots=True)
class MisconceptionRetriever(BM25Retriever):
    """BM25, boosted toward chunks about what the student appears to have got wrong.

    A chunk's `confusions` tag lists the findings it helps distinguish *from*.
    When the student named a category the case does not confirm, chunks that
    address that confusion are the pedagogically useful ones even if another
    chunk matched their words more closely.

    The boost multiplies the raw BM25 score before normalisation, so a chunk
    with no lexical match at all still scores zero and is still dropped. The
    boost reorders; it never conjures relevance from nothing.
    """

    boost: float = MISCONCEPTION_BOOST

    def _rank(self, query: RetrievalQuery) -> list[tuple[str, float]]:
        ranked = self._index.search(query.free_text)
        mistaken = set(query.misconceived_categories)
        if not mistaken:
            return ranked
        by_id = self.corpus.by_id
        adjusted: list[tuple[str, float, int]] = []
        for position, (doc_id, raw) in enumerate(ranked):
            chunk = by_id.get(doc_id)
            multiplier = (
                self.boost
                if chunk is not None and mistaken.intersection(chunk.confusions)
                else 1.0
            )
            adjusted.append((doc_id, raw * multiplier, position))
        adjusted.sort(key=lambda row: (-row[1], row[2]))
        return [(doc_id, value) for doc_id, value, _ in adjusted]
