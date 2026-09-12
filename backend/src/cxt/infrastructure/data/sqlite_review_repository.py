"""SQLAlchemy-backed IReviewRepository."""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from cxt.domain.entities.review import EducatorReview
from cxt.domain.ports.review_repository import IReviewRepository
from cxt.infrastructure.data.database import session_scope
from cxt.infrastructure.data.orm import ReviewRow
from cxt.shared.errors import NotFoundError, ValidationError


def _row_to_review(row: ReviewRow) -> EducatorReview:
    try:
        scores = json.loads(row.scores_json)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"Stored review {row.id} has malformed scores_json."
        ) from exc
    if not isinstance(scores, dict):
        raise ValidationError(f"Stored review {row.id} has non-object scores_json.")
    normalized = {str(k): int(v) for k, v in scores.items()}
    return EducatorReview(
        id=row.id,
        attempt_id=row.attempt_id,
        educator_id=row.educator_id,
        scores=normalized,
        comment=row.comment or "",
        created_at=row.created_at,
    )


class SqliteReviewRepository(IReviewRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._factory = session_factory

    def add(self, review: EducatorReview) -> None:
        scores_payload: dict[str, Any] = {k: int(v) for k, v in review.scores.items()}
        with session_scope(self._factory) as session:
            session.add(
                ReviewRow(
                    id=review.id,
                    attempt_id=review.attempt_id,
                    educator_id=review.educator_id,
                    scores_json=json.dumps(scores_payload, ensure_ascii=False),
                    comment=review.comment,
                    created_at=review.created_at,
                )
            )

    def get(self, review_id: str) -> EducatorReview:
        with session_scope(self._factory) as session:
            row = session.get(ReviewRow, review_id)
            if row is None:
                raise NotFoundError(f"Review not found: {review_id}")
            return _row_to_review(row)

    def list_for_attempt(self, attempt_id: str) -> list[EducatorReview]:
        with session_scope(self._factory) as session:
            stmt = (
                select(ReviewRow)
                .where(ReviewRow.attempt_id == attempt_id)
                .order_by(ReviewRow.created_at.desc())
            )
            return [_row_to_review(row) for row in session.execute(stmt).scalars().all()]
