"""Span-level mention detection, for highlighting a student's answer.

Two properties matter. The offsets must land on the text the student typed,
not on the lower-cased, whitespace-collapsed copy the matching runs on. And
the categories must agree with `named_categories` / `negated_categories`,
because those drive the misconception retrieval arm: a highlight saying one
thing while retrieval acted on another would be two answers to one question.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cxt.infrastructure.retrieval.misconception import (
    find_mentions,
    named_categories,
    negated_categories,
)

_FIXTURES = Path(__file__).resolve().parents[4] / "fixtures" / "student_attempts.json"


def _fixture_texts() -> list[str]:
    data = json.loads(_FIXTURES.read_text(encoding="utf-8"))
    return [
        entry["text"]
        for key, entries in data.items()
        if not key.startswith("_")
        for entry in entries
    ]


_HAND_WRITTEN = [
    "There is a large pleural effusion.",
    "Cardiomegaly with a right pleural effusion.",
    "PNEUMOTHORAX on the left",
    "There is no pneumothorax.",
    "No pneumothorax, but there is a large effusion.",
    "Pneumothorax is ruled out; there is consolidation in the right lower zone.",
    "There is consolidation; pneumothorax is excluded.",
    "No evidence of a mass. The heart is enlarged.",
    "Effusion versus consolidation at the left base.",
    "No effusion on the left, small effusion on the right.",
    "Normal chest. Unremarkable film.",
    "pleural\n   effusion   at the  right base",
]


class TestOffsets:
    def test_span_is_the_text_the_student_typed(self) -> None:
        text = "There is a large Pleural Effusion on the right."
        (mention,) = find_mentions(text)

        assert mention.category == "pleural_effusion"
        assert text[mention.start : mention.end] == "Pleural Effusion"
        assert mention.text == "Pleural Effusion"

    def test_offsets_survive_collapsed_whitespace(self) -> None:
        # Matching runs on "pleural effusion at the right base"; the
        # highlight must still cover the original line break and spaces.
        text = "Large   pleural\n   effusion here, and   cardiomegaly."
        by_category = {m.category: m for m in find_mentions(text)}

        effusion = by_category["pleural_effusion"]
        assert text[effusion.start : effusion.end] == "pleural\n   effusion"
        heart = by_category["cardiomegaly"]
        assert text[heart.start : heart.end] == "cardiomegaly"

    def test_longest_alias_wins_and_spans_do_not_overlap(self) -> None:
        mentions = find_mentions("A small pleural effusion.")

        assert [m.text for m in mentions] == ["pleural effusion"]

    def test_mentions_are_in_reading_order(self) -> None:
        mentions = find_mentions("Cardiomegaly, then an effusion, then a nodule.")

        assert [m.category for m in mentions] == [
            "cardiomegaly",
            "pleural_effusion",
            "nodule_or_mass",
        ]

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_blank_text_has_no_mentions(self, text: str) -> None:
        assert find_mentions(text) == ()


class TestNegation:
    def test_forward_negation(self) -> None:
        (mention,) = find_mentions("There is no pneumothorax.")

        assert mention.category == "pneumothorax"
        assert mention.negated is True

    def test_postposed_negation(self) -> None:
        mentions = {m.category: m for m in find_mentions(
            "Pneumothorax is ruled out; there is consolidation."
        )}

        assert mentions["pneumothorax"].negated is True
        assert mentions["consolidation"].negated is False

    def test_contrast_ends_the_negation(self) -> None:
        mentions = {m.category: m for m in find_mentions(
            "No pneumothorax, but there is a large effusion."
        )}

        assert mentions["pneumothorax"].negated is True
        assert mentions["pleural_effusion"].negated is False

    def test_the_same_finding_can_be_named_and_negated(self) -> None:
        mentions = find_mentions("No effusion on the left; small effusion on the right.")

        assert [(m.category, m.negated) for m in mentions] == [
            ("pleural_effusion", True),
            ("pleural_effusion", False),
        ]

    def test_a_comma_does_not_end_the_negation(self) -> None:
        # Known limitation, inherited from `strip_negated_spans` and kept so
        # the two stay in agreement: the window runs on past a comma, so the
        # second, positive mention is read as negated too.
        mentions = find_mentions("No effusion on the left, small effusion on the right.")

        assert [m.negated for m in mentions] == [True, True]


@pytest.mark.parametrize("text", _HAND_WRITTEN + _fixture_texts())
def test_categories_agree_with_named_and_negated(text: str) -> None:
    mentions = find_mentions(text)
    asserted = {m.category for m in mentions if not m.negated}
    negated = {m.category for m in mentions if m.negated} - asserted

    assert asserted == set(named_categories(text))
    assert negated == set(negated_categories(text))
