"""SQLAlchemy-backed IAttemptRepository.

Replaces the Phase 1 in-memory adapter. Stores the generated `Feedback`
as JSON in `feedback_json` so GETs don't need to re-run the model.
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from cxt.domain.entities.feedback import (
    Feedback,
    RetrievedPassage,
)
from cxt.domain.entities.student_attempt import StudentAttempt
from cxt.domain.ports.attempt_repository import IAttemptRepository
from cxt.domain.value_objects.confidence import Confidence
from cxt.domain.value_objects.finding_label import FindingLabel
from cxt.domain.value_objects.rationale import Rationale
from cxt.infrastructure.data.database import session_scope
from cxt.infrastructure.data.orm import AttemptRow
from cxt.shared.errors import NotFoundError


def _row_to_attempt(row: AttemptRow) -> StudentAttempt:
    return StudentAttempt(
        id=row.id,
        case_id=row.case_id,
        student_id=row.student_id,
        free_text=row.free_text,
        submitted_at=row.submitted_at,
    )


def _row_to_feedback(attempt_id: str, payload: dict[str, Any]) -> Feedback:
    primary = payload.get("primary_finding") or {}
    category_value = primary.get("category") or payload.get("likely_finding") or "no_finding"
    raw_text = payload.get("raw_text") or ""
    return Feedback(
        attempt_id=attempt_id,
        likely_finding=FindingLabel(str(category_value)),
        key_signs=Rationale(str(payload.get("key_signs", ""))),
        comparison=Rationale(str(payload.get("comparison", ""))),
        missed_clues=Rationale(str(payload.get("missed_clues", ""))),
        learning_point=Rationale(str(payload.get("learning_point", ""))),
        confidence=Confidence(float(payload.get("confidence", 0.5))),
        sources=tuple(
            RetrievedPassage(text=p["text"], source=p["source"], score=float(p.get("score", 1.0)))
            for p in payload.get("sources", [])
        ),
        raw={"parsed": payload, "raw_text": raw_text},
    )


def _maybe_load_feedback(attempt_id: str, raw: str | None) -> Feedback | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _row_to_feedback(attempt_id, payload)


def feedback_to_dict(feedback: Feedback) -> dict[str, Any]:
    category = feedback.likely_finding.value
    return {
        "primary_finding": {
            "category": category,
            "label": category,
            "confidence": float(feedback.confidence.value),
            "rationale": feedback.key_signs.value,
        },
        "likely_finding": category,
        "key_signs": feedback.key_signs.value,
        "comparison": feedback.comparison.value,
        "missed_clues": feedback.missed_clues.value,
        "learning_point": feedback.learning_point.value,
        "confidence": float(feedback.confidence.value),
        "sources": [
            {"text": s.text, "source": s.source, "score": float(s.score)}
            for s in feedback.sources
        ],
        "raw_text": (feedback.raw.get("raw_text", "") if isinstance(feedback.raw, dict) else ""),
    }


class SqliteAttemptRepository(IAttemptRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._factory = session_factory

    def add(self, attempt: StudentAttempt) -> None:
        with session_scope(self._factory) as session:
            session.add(
                AttemptRow(
                    id=attempt.id,
                    case_id=attempt.case_id,
                    student_id=attempt.student_id,
                    free_text=attempt.free_text,
                    feedback_json=None,
                    submitted_at=attempt.submitted_at,
                )
            )

    def attach_feedback(self, attempt_id: str, feedback: Feedback) -> None:
        payload = json.dumps(feedback_to_dict(feedback), ensure_ascii=False)
        with session_scope(self._factory) as session:
            row = session.get(AttemptRow, attempt_id)
            if row is None:
                raise NotFoundError(f"Attempt not found: {attempt_id}")
            row.feedback_json = payload

    def get(self, attempt_id: str) -> StudentAttempt:
        with session_scope(self._factory) as session:
            row = session.get(AttemptRow, attempt_id)
            if row is None:
                raise NotFoundError(f"Attempt not found: {attempt_id}")
            return _row_to_attempt(row)

    def get_with_feedback(self, attempt_id: str) -> tuple[StudentAttempt, Feedback | None]:
        with session_scope(self._factory) as session:
            row = session.get(AttemptRow, attempt_id)
            if row is None:
                raise NotFoundError(f"Attempt not found: {attempt_id}")
            attempt = _row_to_attempt(row)
            feedback = _maybe_load_feedback(row.id, row.feedback_json)
            return attempt, feedback

    def list_for_student(self, student_id: str) -> list[StudentAttempt]:
        with session_scope(self._factory) as session:
            stmt = (
                select(AttemptRow)
                .where(AttemptRow.student_id == student_id)
                .order_by(AttemptRow.submitted_at.desc())
            )
            return [_row_to_attempt(row) for row in session.execute(stmt).scalars().all()]

    def list_for_educator_review(self) -> list[StudentAttempt]:
        with session_scope(self._factory) as session:
            stmt = select(AttemptRow).order_by(AttemptRow.submitted_at.desc())
            return [_row_to_attempt(row) for row in session.execute(stmt).scalars().all()]
