"""Alias matching, and the negation bug that makes it dangerous.

"There is no pneumothorax" contains the word `pneumothorax`. Matched naively, a
student who correctly excluded a pneumothorax is recorded as having claimed
one, and the misconception arm then retrieves pneumothorax material to correct
an error they did not make — the system confidently teaching against a correct
answer. Half this file is that one case.
"""
from __future__ import annotations

import pytest

from cxt.infrastructure.retrieval.misconception import (
    CATEGORY_ALIASES,
    mentions_category,
    named_categories,
    negated_categories,
    strip_negated_spans,
)


class TestPlainMatching:
    def test_finds_a_named_category(self) -> None:
        assert named_categories("There is a large pleural effusion.") == (
            "pleural_effusion",
        )

    def test_finds_several(self) -> None:
        found = named_categories("Cardiomegaly with a right pleural effusion.")
        assert set(found) == {"cardiomegaly", "pleural_effusion"}

    def test_is_case_insensitive(self) -> None:
        assert named_categories("PNEUMOTHORAX on the left") == ("pneumothorax",)

    def test_matches_colloquial_aliases(self) -> None:
        assert named_categories("the lung looks collapsed — lung collapse") == (
            "pneumothorax",
        )

    def test_empty_text_names_nothing(self) -> None:
        assert named_categories("") == ()
        assert named_categories("   ") == ()

    def test_unrelated_text_names_nothing(self) -> None:
        assert named_categories("I am not sure what I am looking at.") == ()

    def test_word_boundaries_are_respected(self) -> None:
        """"massive" must not match the `mass` alias."""
        assert "nodule_or_mass" not in named_categories("There is massive scarring.")

    def test_order_is_stable_regardless_of_phrasing(self) -> None:
        a = named_categories("Effusion and cardiomegaly.")
        b = named_categories("Cardiomegaly and effusion.")
        assert a == b


class TestNegation:
    """The bug this module exists to avoid."""

    def test_simple_negation_does_not_count_as_naming(self) -> None:
        assert named_categories("There is no pneumothorax.") == ()

    def test_negated_category_is_reported_separately(self) -> None:
        assert negated_categories("There is no pneumothorax.") == ("pneumothorax",)

    @pytest.mark.parametrize(
        "text",
        [
            "There is no pneumothorax.",
            "No pneumothorax is seen.",
            "No evidence of pneumothorax.",
            "No obvious pneumothorax.",
            "I do not see a pneumothorax.",
            "The film is free of pneumothorax.",
            "Pneumothorax is ruled out.",
            "There are no pneumothoraces.",
        ],
    )
    def test_negation_phrasings(self, text: str) -> None:
        assert "pneumothorax" not in named_categories(text)

    def test_negation_does_not_swallow_the_following_clause(self) -> None:
        """The whole point: one finding excluded, another asserted."""
        found = named_categories(
            "There is no pneumothorax, but there is a large pleural effusion."
        )
        assert found == ("pleural_effusion",)

    def test_negation_stops_at_a_full_stop(self) -> None:
        found = named_categories("No pneumothorax. There is consolidation.")
        assert found == ("consolidation",)

    def test_negation_stops_at_a_semicolon(self) -> None:
        found = named_categories("No pneumothorax; cardiomegaly is present.")
        assert found == ("cardiomegaly",)

    def test_a_distant_mention_is_still_counted(self) -> None:
        """Negation is scoped to a clause, not the rest of the answer."""
        text = (
            "No pneumothorax is present on this film. The heart is enlarged and "
            "the cardiothoracic ratio is clearly increased, so I would call this "
            "cardiomegaly."
        )
        assert "cardiomegaly" in named_categories(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Pneumothorax is ruled out.",
            "Pneumothorax is excluded.",
            "A pneumothorax is absent.",
            "Pneumothorax is not present.",
            "Pneumothorax seems unlikely.",
        ],
    )
    def test_negation_after_the_finding(self, text: str) -> None:
        """English negates on either side. A forward-only scanner gets this
        exactly backwards and records the student as claiming what they ruled out."""
        assert "pneumothorax" not in named_categories(text)
        assert "pneumothorax" in negated_categories(text)

    def test_postposed_negation_does_not_reach_a_previous_clause(self) -> None:
        found = named_categories(
            "There is consolidation at the right base; pneumothorax is excluded."
        )
        assert found == ("consolidation",)

    def test_strip_preserves_length(self) -> None:
        """Spans are blanked, not deleted, so adjacent clauses cannot fuse into
        a phrase nobody wrote."""
        text = "There is no pneumothorax but an effusion."
        assert len(strip_negated_spans(text)) == len(text.lower())

    def test_a_correct_exclusion_is_not_treated_as_an_error(self) -> None:
        """End-to-end statement of the failure being prevented."""
        text = "No pneumothorax. The right base shows consolidation."
        assert "pneumothorax" not in named_categories(text)
        assert "pneumothorax" in negated_categories(text)


class TestMisconceivedCategories:
    def test_named_but_unconfirmed_is_the_misconception(self) -> None:
        from cxt.domain.ports.retriever import RetrievalQuery

        query = RetrievalQuery(
            free_text="I think this is a pleural effusion.",
            confirmed_categories=("consolidation",),
            named_categories=("pleural_effusion",),
        )
        assert query.misconceived_categories == ("pleural_effusion",)

    def test_a_correct_answer_yields_no_misconception(self) -> None:
        from cxt.domain.ports.retriever import RetrievalQuery

        query = RetrievalQuery(
            free_text="Consolidation at the right base.",
            confirmed_categories=("consolidation",),
            named_categories=("consolidation",),
        )
        assert query.misconceived_categories == ()


class TestAliasTableHygiene:
    def test_every_category_has_aliases(self) -> None:
        for category, aliases in CATEGORY_ALIASES.items():
            assert aliases, category

    def test_aliases_are_lowercase(self) -> None:
        """Matching lowercases the haystack, so an uppercase alias never fires."""
        for aliases in CATEGORY_ALIASES.values():
            for alias in aliases:
                assert alias == alias.lower()

    def test_mentions_category_ignores_negation_by_design(self) -> None:
        """`mentions_category` is the raw primitive; `named_categories` is the
        one that handles negation. Keeping them distinct is deliberate."""
        assert mentions_category("no pneumothorax", "pneumothorax")
        assert named_categories("no pneumothorax") == ()
