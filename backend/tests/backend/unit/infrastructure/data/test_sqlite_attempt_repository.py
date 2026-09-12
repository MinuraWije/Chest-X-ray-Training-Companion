"""Tests for SqliteAttemptRepository."""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from cxt.domain.entities.feedback import Feedback
from cxt.domain.entities.student_attempt import StudentAttempt
from cxt.domain.value_objects.confidence import Confidence
from cxt.domain.value_objects.finding_label import FindingLabel
from cxt.domain.value_objects.rationale import Rationale
from cxt.infrastructure.config.settings import Settings
from cxt.infrastructure.data.database import (
    build_engine,
    build_session_factory,
    init_database,
)
from cxt.infrastructure.data.sqlite_attempt_repository import (
    SqliteAttemptRepository,
    _row_to_feedback,
    feedback_to_dict,
)
from cxt.shared.errors import NotFoundError


@pytest.fixture()
def attempts(tmp_path: Path) -> Iterator[SqliteAttemptRepository]:
    settings = Settings(database_url=f"sqlite+pysqlite:///{tmp_path / 'cxt.db'}")
    engine = build_engine(settings)
    init_database(engine)
    factory = build_session_factory(engine)
    yield SqliteAttemptRepository(factory)


def _new_attempt(id_: str = "a-1", student_id: str = "u-1") -> StudentAttempt:
    return StudentAttempt.new(
        id=id_,
        case_id="c-1",
        student_id=student_id,
        free_text="interpretation",
    )


def _make_feedback(attempt_id: str) -> Feedback:
    return Feedback(
        attempt_id=attempt_id,
        likely_finding=FindingLabel("pneumothorax"),
        key_signs=Rationale("Visible visceral pleural line."),
        comparison=Rationale("Matches the case."),
        missed_clues=Rationale("None."),
        learning_point=Rationale("Look at the apices."),
        confidence=Confidence(0.85),
    )


class TestAddAndGet:
    def test_round_trip(self, attempts: SqliteAttemptRepository) -> None:
        attempts.add(_new_attempt())
        a = attempts.get("a-1")
        assert a.case_id == "c-1"
        assert a.student_id == "u-1"

    def test_get_unknown_raises(self, attempts: SqliteAttemptRepository) -> None:
        with pytest.raises(NotFoundError):
            attempts.get("nope")


class TestListForStudent:
    def test_filters(self, attempts: SqliteAttemptRepository) -> None:
        attempts.add(_new_attempt("a-1", student_id="u-1"))
        attempts.add(_new_attempt("a-2", student_id="u-2"))
        results = attempts.list_for_student("u-1")
        assert [a.id for a in results] == ["a-1"]

    def test_empty(self, attempts: SqliteAttemptRepository) -> None:
        assert attempts.list_for_student("nobody") == []


class TestListForEducator:
    def test_returns_all_sorted_desc(self, attempts: SqliteAttemptRepository) -> None:
        attempts.add(_new_attempt("a-1"))
        attempts.add(_new_attempt("a-2"))
        results = attempts.list_for_educator_review()
        assert {a.id for a in results} == {"a-1", "a-2"}


class TestAttachFeedback:
    def test_attach_and_get_with_feedback(self, attempts: SqliteAttemptRepository) -> None:
        attempts.add(_new_attempt())
        attempts.attach_feedback("a-1", _make_feedback("a-1"))
        attempt, feedback = attempts.get_with_feedback("a-1")
        assert attempt.id == "a-1"
        assert feedback is not None
        assert feedback.likely_finding.value == "pneumothorax"
        assert feedback.confidence.value == pytest.approx(0.85)

    def test_attach_to_unknown_raises(self, attempts: SqliteAttemptRepository) -> None:
        with pytest.raises(NotFoundError):
            attempts.attach_feedback("nope", _make_feedback("nope"))

    def test_get_with_feedback_returns_none_when_no_feedback(
        self, attempts: SqliteAttemptRepository
    ) -> None:
        attempts.add(_new_attempt())
        _attempt, feedback = attempts.get_with_feedback("a-1")
        assert feedback is None


def test_feedback_to_dict_round_trip() -> None:
    original = _make_feedback("a-1")
    payload = feedback_to_dict(original)
    encoded = json.dumps(payload, ensure_ascii=False)
    decoded = json.loads(encoded)
    reconstructed = _row_to_feedback("a-1", decoded)
    assert reconstructed.likely_finding.value == original.likely_finding.value
    assert reconstructed.confidence.value == original.confidence.value
    assert reconstructed.key_signs.value == original.key_signs.value
    assert reconstructed.learning_point.value == original.learning_point.value
