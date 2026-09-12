"""Okapi BM25 ranking, written before the implementation.

This file exists because a BM25 bug is silent. Get the length normalisation
wrong and the ranker degenerates toward "return the longest chunk" — it still
returns plausible medical prose, still scores, still renders in the UI, and
every retrieval number in the Phase 2 report is quietly meaningless. There is
no output you could eyeball to catch it. So the expected ordering is asserted
against a toy corpus small enough to reason about by hand.

The toy corpus is deliberately not radiology: made-up words make the term
statistics obvious, and stop anyone reading a result as domain knowledge.
"""
from __future__ import annotations

import math

import pytest

from cxt.infrastructure.retrieval.bm25 import BM25Index, normalise_score, tokenize

# doc id -> text. Term frequencies chosen so the ranking is hand-checkable.
TOY_CORPUS: dict[str, str] = {
    # "zeta" appears here twice and nowhere else: the strongest possible signal.
    "a": "zeta zeta alpha",
    # "zeta" once, plus filler that makes this document long.
    "b": "zeta alpha beta gamma delta epsilon omega sigma tau upsilon phi chi",
    # No "zeta" at all.
    "c": "alpha beta gamma",
    # "alpha" in every document, so it carries almost no discriminating power.
    "d": "alpha",
}


@pytest.fixture()
def index() -> BM25Index:
    return BM25Index.build(TOY_CORPUS)


class TestTokenize:
    def test_lowercases_and_splits_on_non_letters(self) -> None:
        assert tokenize("Pleural-effusion, LEFT.") == ["pleural", "effusion", "left"]

    def test_keeps_digits(self) -> None:
        """Cardiothoracic ratios and "0.50" are meaningful corpus terms."""
        assert "0" in tokenize("ratio 0 5") or "5" in tokenize("ratio 0 5")

    def test_empty_input_yields_no_tokens(self) -> None:
        assert tokenize("") == []
        assert tokenize("   ,.!  ") == []


class TestRanking:
    def test_higher_term_frequency_outranks_lower(self, index: BM25Index) -> None:
        """`a` has zeta twice, `b` once. `a` must come first."""
        ranked = index.search("zeta")
        assert [doc_id for doc_id, _ in ranked] == ["a", "b"]

    def test_documents_without_the_term_are_not_returned(self, index: BM25Index) -> None:
        returned = {doc_id for doc_id, _ in index.search("zeta")}
        assert "c" not in returned
        assert "d" not in returned

    def test_length_normalisation_does_not_favour_long_documents(
        self, index: BM25Index
    ) -> None:
        """The bug this whole file exists to catch.

        `b` is four times the length of `a` and contains `zeta` half as often.
        A broken normalisation — dividing by the wrong length, or omitting the
        average — inverts this, and nothing downstream would show it.
        """
        scores = dict(index.search("zeta"))
        assert scores["a"] > scores["b"]

    def test_a_rare_term_outweighs_a_ubiquitous_one(self, index: BM25Index) -> None:
        """`alpha` is in all four documents; its IDF should be near zero."""
        ranked = index.search("zeta alpha")
        assert ranked[0][0] == "a"
        # `d` is *only* alpha and maximally short, so a missing IDF term would
        # float it to the top on term-frequency-times-brevity alone.
        assert ranked[0][1] > dict(ranked).get("d", 0.0)

    def test_repeated_query_terms_do_not_change_the_order(self, index: BM25Index) -> None:
        assert [d for d, _ in index.search("zeta zeta")] == [
            d for d, _ in index.search("zeta")
        ]

    def test_results_are_sorted_descending(self, index: BM25Index) -> None:
        scores = [s for _, s in index.search("alpha zeta beta")]
        assert scores == sorted(scores, reverse=True)

    def test_ties_break_deterministically(self) -> None:
        """Two runs of the same query must give the same order, or the eval
        measures run-to-run noise and calls it retrieval quality."""
        index = BM25Index.build({"x": "same words here", "y": "same words here"})
        assert index.search("same words") == index.search("same words")


class TestDegenerateInputs:
    def test_empty_query_returns_nothing(self, index: BM25Index) -> None:
        assert index.search("") == []
        assert index.search("   ") == []

    def test_query_with_no_matching_term_returns_nothing(self, index: BM25Index) -> None:
        assert index.search("nonexistentterm") == []

    def test_empty_corpus_is_searchable_and_empty(self) -> None:
        index = BM25Index.build({})
        assert index.search("anything") == []
        assert index.size == 0

    def test_single_document_corpus_does_not_divide_by_zero(self) -> None:
        """With N=1 the classic IDF numerator goes to zero or negative."""
        index = BM25Index.build({"only": "solitary document"})
        results = index.search("solitary")
        assert len(results) == 1
        assert results[0][1] > 0.0

    def test_documents_with_no_tokens_are_ignored_not_crashed_on(self) -> None:
        index = BM25Index.build({"blank": "   ", "real": "content here"})
        assert [d for d, _ in index.search("content")] == ["real"]


class TestNormalisation:
    """`score = bm25 / (bm25 + k)`: saturating, absolute, and thresholdable.

    Not top-1 normalisation. That makes the best hit 1.0 even when BM25 gave it
    almost nothing, so the UI would print "100%" beside a garbage match in a
    medical teaching tool — and it makes an absolute abstention floor
    impossible, because the top hit is always 1.0 by construction.
    """

    def test_bounded_in_unit_interval(self) -> None:
        for raw in (0.0, 0.001, 1.0, 12.5, 1e6):
            assert 0.0 <= normalise_score(raw) < 1.0

    def test_zero_maps_to_zero(self) -> None:
        assert normalise_score(0.0) == 0.0

    def test_monotone_in_the_raw_score(self) -> None:
        values = [normalise_score(r) for r in (0.5, 1.0, 2.0, 4.0, 8.0)]
        assert values == sorted(values)

    def test_saturates_rather_than_reaching_one(self) -> None:
        assert normalise_score(1e9) < 1.0
        assert math.isclose(normalise_score(1e9), 1.0, abs_tol=1e-6)

    def test_never_exceeds_the_retrieved_passage_bound(self) -> None:
        """RetrievedPassage validates score in [0,1]; that check is load-bearing."""
        assert normalise_score(float("inf")) <= 1.0

    def test_negative_raw_score_clamps_to_zero(self) -> None:
        """BM25 IDF can go negative for a term in more than half the corpus."""
        assert normalise_score(-3.0) == 0.0
