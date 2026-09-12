"""The four retrieval arms, against a hand-built corpus.

Each arm's distinguishing behaviour is asserted, because the Phase 2 report
compares them and a bug that made two arms behave identically would show up as
"retrieval made no difference" rather than as a defect.
"""
from __future__ import annotations

import pytest

from cxt.domain.ports.retriever import RetrievalQuery
from cxt.infrastructure.retrieval.corpus import Corpus, CorpusChunk
from cxt.infrastructure.retrieval.retrievers import (
    BM25Retriever,
    CategoryRetriever,
    MisconceptionRetriever,
    NullRetriever,
)


def _chunk(
    chunk_id: str,
    text: str,
    findings: tuple[str, ...],
    confusions: tuple[str, ...] = (),
    aspect: str = "radiographic_sign",
) -> CorpusChunk:
    return CorpusChunk(
        id=chunk_id,
        title=chunk_id.replace("-", " "),
        text=text,
        source="Test source",
        url="https://example.org/doc",
        licence="CC BY-NC-SA 3.0",
        licence_url="https://example.org/l",
        retrieved_on="2026-09-12",
        findings=findings,
        confusions=confusions,
        aspect=aspect,
    )


@pytest.fixture()
def corpus() -> Corpus:
    return Corpus(
        version="test-corpus",
        chunks=(
            _chunk(
                "effusion-meniscus",
                "Pleural fluid forms a concave meniscus rising laterally.",
                ("pleural_effusion",),
            ),
            _chunk(
                "effusion-vs-consolidation",
                "Distinguishing an effusion from consolidation at the base.",
                ("pleural_effusion", "consolidation"),
                confusions=("pleural_effusion", "consolidation"),
                aspect="differential",
            ),
            _chunk(
                "pneumothorax-line",
                "A thin visceral pleural line with absent markings beyond it.",
                ("pneumothorax",),
            ),
            _chunk(
                "drainage",
                "Chest drain insertion technique for a large effusion.",
                ("pleural_effusion",),
                aspect="management",
            ),
        ),
    )


class TestNullRetriever:
    def test_returns_nothing(self) -> None:
        assert NullRetriever().retrieve(query=RetrievalQuery(free_text="anything")) == ()


class TestBM25Retriever:
    def test_matches_on_the_students_words(self, corpus: Corpus) -> None:
        results = BM25Retriever(corpus=corpus).retrieve(
            query=RetrievalQuery(free_text="a concave meniscus of pleural fluid")
        )
        assert results[0].id == "effusion-meniscus"

    def test_ignores_the_answer_key(self, corpus: Corpus) -> None:
        """The arm's whole value is that nothing about the confirmed finding
        participates in ranking, which is what makes its relevance number real."""
        retriever = BM25Retriever(corpus=corpus)
        text = "a thin visceral pleural line"
        without = retriever.retrieve(query=RetrievalQuery(free_text=text))
        with_key = retriever.retrieve(
            query=RetrievalQuery(free_text=text, confirmed_categories=("pleural_effusion",))
        )
        assert [p.id for p in without] == [p.id for p in with_key]

    def test_respects_top_k(self, corpus: Corpus) -> None:
        results = BM25Retriever(corpus=corpus, score_floor=0.0).retrieve(
            query=RetrievalQuery(free_text="pleural effusion consolidation line"), top_k=2
        )
        assert len(results) <= 2

    def test_abstains_below_the_score_floor(self, corpus: Corpus) -> None:
        """Weak matches are withheld, not shown for completeness. A barely
        relevant citation in a medical teaching tool teaches students to
        discount citations."""
        retriever = BM25Retriever(corpus=corpus, score_floor=0.99)
        assert retriever.retrieve(query=RetrievalQuery(free_text="pleural fluid")) == ()

    def test_unmatched_query_returns_nothing(self, corpus: Corpus) -> None:
        results = BM25Retriever(corpus=corpus).retrieve(
            query=RetrievalQuery(free_text="quantum chromodynamics")
        )
        assert results == ()

    def test_empty_free_text_returns_nothing(self, corpus: Corpus) -> None:
        assert BM25Retriever(corpus=corpus).retrieve(query=RetrievalQuery()) == ()

    def test_scores_are_within_the_passage_bound(self, corpus: Corpus) -> None:
        for passage in BM25Retriever(corpus=corpus, score_floor=0.0).retrieve(
            query=RetrievalQuery(free_text="pleural effusion meniscus"), top_k=4
        ):
            assert 0.0 <= passage.score <= 1.0

    def test_passage_carries_id_and_url_for_citation(self, corpus: Corpus) -> None:
        passage = BM25Retriever(corpus=corpus).retrieve(
            query=RetrievalQuery(free_text="concave meniscus")
        )[0]
        assert passage.id == "effusion-meniscus"
        assert passage.url.startswith("http")


class TestCategoryRetriever:
    def test_filters_to_the_confirmed_category(self, corpus: Corpus) -> None:
        results = CategoryRetriever(corpus=corpus).retrieve(
            query=RetrievalQuery(
                free_text="I am not sure", confirmed_categories=("pneumothorax",)
            )
        )
        assert [p.id for p in results] == ["pneumothorax-line"]

    def test_returns_nothing_without_a_confirmed_category(self, corpus: Corpus) -> None:
        results = CategoryRetriever(corpus=corpus).retrieve(
            query=RetrievalQuery(free_text="meniscus")
        )
        assert results == ()

    def test_ranks_within_the_filtered_set_by_the_students_words(
        self, corpus: Corpus
    ) -> None:
        results = CategoryRetriever(corpus=corpus).retrieve(
            query=RetrievalQuery(
                free_text="a concave meniscus", confirmed_categories=("pleural_effusion",)
            )
        )
        assert results[0].id == "effusion-meniscus"

    def test_still_returns_something_when_the_text_matches_nothing(
        self, corpus: Corpus
    ) -> None:
        """The oracle is an upper bound; it should not abstain just because the
        student wrote something unhelpful."""
        results = CategoryRetriever(corpus=corpus).retrieve(
            query=RetrievalQuery(
                free_text="zzzz", confirmed_categories=("pleural_effusion",)
            )
        )
        assert results


class TestMisconceptionRetriever:
    def test_boosts_a_chunk_addressing_the_students_error(self, corpus: Corpus) -> None:
        """The student said effusion; the case is consolidation. The chunk that
        distinguishes exactly those two should rise.

        Asserted as a rank improvement, not as reaching first place. The boost
        is a tie-breaker within a band of comparable scores — it deliberately
        cannot overtake a much stronger lexical match, because a pedagogical
        hint that overrode the evidence would be retrieving on the answer key
        rather than on what the student wrote.
        """
        query = RetrievalQuery(
            free_text="pleural effusion with a concave meniscus at the base",
            confirmed_categories=("consolidation",),
            named_categories=("pleural_effusion",),
        )
        plain = [
            p.id
            for p in BM25Retriever(corpus=corpus, score_floor=0.0).retrieve(
                query=query, top_k=4
            )
        ]
        boosted = [
            p.id
            for p in MisconceptionRetriever(corpus=corpus, score_floor=0.0).retrieve(
                query=query, top_k=4
            )
        ]
        target = "effusion-vs-consolidation"
        assert boosted.index(target) < plain.index(target)

    def test_boost_raises_the_score_of_a_matching_chunk(self, corpus: Corpus) -> None:
        query = RetrievalQuery(
            free_text="distinguishing effusion from consolidation",
            confirmed_categories=("consolidation",),
            named_categories=("pleural_effusion",),
        )
        target = "effusion-vs-consolidation"
        plain = {p.id: p.score for p in BM25Retriever(corpus=corpus).retrieve(query=query)}
        boosted = {
            p.id: p.score for p in MisconceptionRetriever(corpus=corpus).retrieve(query=query)
        }
        assert boosted[target] > plain[target]

    def test_no_misconception_leaves_the_ranking_alone(self, corpus: Corpus) -> None:
        query = RetrievalQuery(
            free_text="a concave meniscus of pleural fluid",
            confirmed_categories=("pleural_effusion",),
            named_categories=("pleural_effusion",),
        )
        plain = BM25Retriever(corpus=corpus).retrieve(query=query)
        boosted = MisconceptionRetriever(corpus=corpus).retrieve(query=query)
        assert [p.id for p in plain] == [p.id for p in boosted]

    def test_boost_cannot_conjure_relevance_from_nothing(self, corpus: Corpus) -> None:
        """A chunk with no lexical match still scores zero. The boost reorders
        within what BM25 found; it never introduces an unmatched chunk."""
        query = RetrievalQuery(
            free_text="quantum chromodynamics",
            confirmed_categories=("consolidation",),
            named_categories=("pleural_effusion",),
        )
        assert MisconceptionRetriever(corpus=corpus).retrieve(query=query) == ()

    def test_still_honours_the_score_floor(self, corpus: Corpus) -> None:
        query = RetrievalQuery(
            free_text="pleural fluid",
            confirmed_categories=("consolidation",),
            named_categories=("pleural_effusion",),
        )
        retriever = MisconceptionRetriever(corpus=corpus, score_floor=0.99)
        assert retriever.retrieve(query=query) == ()
