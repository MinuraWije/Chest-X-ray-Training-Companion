"""Educator review routes: list the queue, fetch an attempt for review, submit a review.

Routes:
- `GET  /api/v1/reviews/queue`                      list attempts awaiting review
- `GET  /api/v1/reviews/queue/{attempt_id}`         fetch the attempt with feedback
- `GET  /api/v1/reviews/attempts/{attempt_id}`      list reviews for one attempt
- `POST /api/v1/reviews`                            submit a review
"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel, Field

from cxt.application.use_cases.reviews import (
    ListReviewQueueUseCase,
    ListReviewsForAttemptUseCase,
    ReviewAttemptRequest,
    ReviewAttemptUseCase,
)
from cxt.domain.entities.review import EducatorReview
from cxt.domain.entities.user import User
from cxt.domain.ports.attempt_repository import IAttemptRepository
from cxt.domain.ports.review_repository import IReviewRepository
from cxt.interfaces.api.dependencies import (
    get_attempt_repository,
    get_review_repository,
    require_role,
)
from cxt.interfaces.api.schemas.feedback import shape_feedback
from cxt.shared.errors import ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/reviews", tags=["reviews"])


class ReviewSubmitBody(BaseModel):
    attemptId: str = Field(min_length=1, max_length=64)
    criteriaScores: dict[str, int] = Field(min_length=1)
    comments: str = Field(default="", max_length=4000)


class ReviewDto(BaseModel):
    id: str
    attemptId: str
    educatorId: str
    criteriaScores: dict[str, int]
    comments: str
    createdAt: str


class AttemptSummaryDto(BaseModel):
    id: str
    caseId: str
    studentId: str
    freeText: str
    submittedAt: str


def _attempt_to_summary(attempt) -> AttemptSummaryDto:  # type: ignore[no-untyped-def]
    return AttemptSummaryDto(
        id=attempt.id,
        caseId=attempt.case_id,
        studentId=attempt.student_id,
        freeText=attempt.free_text,
        submittedAt=attempt.submitted_at.isoformat(),
    )


def _review_to_dto(review: EducatorReview) -> ReviewDto:
    return ReviewDto(
        id=review.id,
        attemptId=review.attempt_id,
        educatorId=review.educator_id,
        criteriaScores=dict(review.scores),
        comments=review.comment,
        createdAt=review.created_at.isoformat(),
    )


def _list_queue_use_case(
    attempts: IAttemptRepository = Depends(get_attempt_repository),
) -> ListReviewQueueUseCase:
    return ListReviewQueueUseCase(attempt_repository=attempts)


def _review_use_case(
    reviews: IReviewRepository = Depends(get_review_repository),
    attempts: IAttemptRepository = Depends(get_attempt_repository),
) -> ReviewAttemptUseCase:
    return ReviewAttemptUseCase(review_repository=reviews, attempt_repository=attempts)


def _list_reviews_use_case(
    reviews: IReviewRepository = Depends(get_review_repository),
) -> ListReviewsForAttemptUseCase:
    return ListReviewsForAttemptUseCase(review_repository=reviews)


@router.get("/queue")
def list_queue(
    user: User = Depends(require_role("EDUCATOR", "ADMIN")),
    use_case: ListReviewQueueUseCase = Depends(_list_queue_use_case),
) -> dict[str, object]:
    items = use_case.execute(caller_role=user.role)
    return {
        "items": [_attempt_to_summary(a).model_dump() for a in items],
        "total": len(items),
    }


@router.get("/queue/{attempt_id}")
def get_queue_item(
    attempt_id: Annotated[str, Path(min_length=1, max_length=64)],
    user: User = Depends(require_role("EDUCATOR", "ADMIN")),
    attempts: IAttemptRepository = Depends(get_attempt_repository),
) -> dict[str, object]:
    attempt, feedback = attempts.get_with_feedback(attempt_id)
    return _shape_attempt_with_feedback(attempt, feedback)


@router.get("/attempts/{attempt_id}")
def list_reviews_for_attempt(
    attempt_id: Annotated[str, Path(min_length=1, max_length=64)],
    user: User = Depends(require_role("EDUCATOR", "ADMIN")),
    use_case: ListReviewsForAttemptUseCase = Depends(_list_reviews_use_case),
) -> dict[str, object]:
    items = use_case.execute(attempt_id=attempt_id, caller_role=user.role)
    return {
        "items": [_review_to_dto(r).model_dump() for r in items],
        "total": len(items),
    }


@router.post("", response_model=ReviewDto)
def submit_review(
    body: ReviewSubmitBody,
    user: User = Depends(require_role("EDUCATOR", "ADMIN")),
    use_case: ReviewAttemptUseCase = Depends(_review_use_case),
) -> ReviewDto:
    for criterion, score in body.criteriaScores.items():
        if not isinstance(score, int) or not 1 <= score <= 5:
            raise ValidationError(
                f"criteriaScores[{criterion!r}] must be an int in 1..5, got {score!r}."
            )
    review = use_case.execute(
        ReviewAttemptRequest(
            attempt_id=body.attemptId,
            scores=body.criteriaScores,
            comment=body.comments,
            educator_id=user.id,
            educator_role=user.role,
        )
    )
    return _review_to_dto(review)


def _shape_attempt_with_feedback(attempt, feedback) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """The educator sees exactly the feedback the student saw — same shaper, no copy."""
    summary = _attempt_to_summary(attempt).model_dump()
    summary["feedback"] = shape_feedback(feedback) if feedback is not None else None
    return summary


__all__ = ["router"]
