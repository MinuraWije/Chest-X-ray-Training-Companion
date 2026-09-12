from __future__ import annotations

import pytest

from cxt.domain.entities.feedback import Feedback, FeedbackSection, RetrievedPassage
from cxt.domain.value_objects.confidence import Confidence
from cxt.domain.value_objects.finding_label import FindingLabel
from cxt.domain.value_objects.rationale import Rationale
from cxt.shared.errors import ValidationError


def _good_feedback(**overrides: object) -> Feedback:
    base: dict[str, object] = dict(
        attempt_id="a1",
        likely_finding=FindingLabel("Pneumothorax"),
        key_signs=Rationale("Visible visceral pleural line."),
        comparison=Rationale("Student said pneumothorax; correct."),
        missed_clues=Rationale("Slight mediastinal shift not mentioned."),
        learning_point=Rationale("Always check for mediastinal shift in tension pneumothorax."),
        confidence=Confidence(0.85),
    )
    base.update(overrides)
    return Feedback(**base)  # type: ignore[arg-type]


def test_feedback_happy_path() -> None:
    f = _good_feedback()
    assert f.attempt_id == "a1"
    assert f.sources == ()
    assert f.raw == {}


def test_feedback_accepts_sources_and_raw() -> None:
    src = RetrievedPassage(text="Pneumothorax info", source="textbook", score=0.9)
    f = _good_feedback(sources=(src,), raw={"a": 1})
    assert f.sources == (src,)
    assert f.raw == {"a": 1}


def test_feedback_rejects_bad_attempt_id() -> None:
    with pytest.raises(ValidationError):
        _good_feedback(attempt_id="")


def test_feedback_rejects_non_value_object_fields() -> None:
    with pytest.raises(ValidationError):
        _good_feedback(likely_finding="oops")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        _good_feedback(key_signs="oops")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        _good_feedback(confidence=2.0)  # type: ignore[arg-type]


def test_feedback_section_validation() -> None:
    s = FeedbackSection(heading="Key signs", body="Trachea midline.")
    assert s.heading == "Key signs"
    with pytest.raises(ValidationError):
        FeedbackSection(heading="", body="x")
    with pytest.raises(ValidationError):
        FeedbackSection(heading="x", body="")


def test_retrieved_passage_validation() -> None:
    p = RetrievedPassage(text="t", source="s")
    assert p.score == 1.0
    with pytest.raises(ValidationError):
        RetrievedPassage(text="", source="s")
    with pytest.raises(ValidationError):
        RetrievedPassage(text="t", source="")
    with pytest.raises(ValidationError):
        RetrievedPassage(text="t", source="s", score=1.5)
