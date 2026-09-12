from __future__ import annotations

import dataclasses
import datetime

import pytest

from cxt.domain.entities.student_attempt import StudentAttempt
from cxt.shared.errors import ValidationError


def test_attempt_new_trims_and_stamps() -> None:
    a = StudentAttempt.new(id="a1", case_id="c1", student_id="s1", free_text="  Pneumothorax likely.  ")
    assert a.id == "a1"
    assert a.case_id == "c1"
    assert a.student_id == "s1"
    assert a.free_text == "Pneumothorax likely."
    assert a.submitted_at is not None


def test_attempt_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        StudentAttempt.new(id="a1", case_id="c1", student_id="s1", free_text="   ")


def test_attempt_rejects_empty_ids() -> None:
    now = datetime.datetime.now(datetime.UTC)
    with pytest.raises(ValidationError):
        StudentAttempt(id="a1", case_id="", student_id="s1", free_text="x", submitted_at=now)
    with pytest.raises(ValidationError):
        StudentAttempt(id="", case_id="c1", student_id="s1", free_text="x", submitted_at=now)


def test_attempt_is_frozen() -> None:
    a = StudentAttempt.new(id="a1", case_id="c1", student_id="s1", free_text="x")
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        a.free_text = "y"  # type: ignore[misc]
