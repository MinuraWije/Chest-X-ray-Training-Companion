from __future__ import annotations

import pytest

from cxt.domain.entities.review import EducatorReview
from cxt.shared.errors import ValidationError


def _good_scores() -> dict[str, int]:
    return {
        "finding_recognition": 4,
        "clarity": 5,
        "usefulness_for_learning": 4,
    }


def test_review_new_computes_average() -> None:
    r = EducatorReview.new(
        id="r1",
        attempt_id="a1",
        educator_id="e1",
        scores=_good_scores(),
        comment="  Good attempt.  ",
    )
    assert r.id == "r1"
    assert r.comment == "Good attempt."
    assert r.average_score == pytest.approx((4 + 5 + 4) / 3)


def test_review_rejects_empty_scores() -> None:
    with pytest.raises(ValidationError):
        EducatorReview.new(id="r1", attempt_id="a1", educator_id="e1", scores={}, comment="")


def test_review_rejects_unknown_criterion() -> None:
    with pytest.raises(ValidationError):
        EducatorReview.new(
            id="r1",
            attempt_id="a1",
            educator_id="e1",
            scores={"made_up_criterion": 3},
            comment="",
        )


def test_review_rejects_out_of_range_score() -> None:
    with pytest.raises(ValidationError):
        EducatorReview.new(
            id="r1",
            attempt_id="a1",
            educator_id="e1",
            scores={"clarity": 6},
            comment="",
        )
    with pytest.raises(ValidationError):
        EducatorReview.new(
            id="r1",
            attempt_id="a1",
            educator_id="e1",
            scores={"clarity": 0},
            comment="",
        )


def test_review_rejects_non_int_score() -> None:
    with pytest.raises(ValidationError):
        EducatorReview.new(
            id="r1",
            attempt_id="a1",
            educator_id="e1",
            scores={"clarity": 3.0},  # type: ignore[dict-item]
            comment="",
        )
