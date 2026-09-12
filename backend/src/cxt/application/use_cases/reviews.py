"""Review use cases: list the queue, submit a review.

Both require the caller to be an EDUCATOR or ADMIN (enforced here so the
use case is safe to call from any delivery mechanism, not just HTTP).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from cxt.domain.entities.review import EducatorReview
from cxt.domain.entities.role import Role
from cxt.domain.entities.student_attempt import StudentAttempt
from cxt.domain.ports.attempt_repository import IAttemptRepository
from cxt.domain.ports.review_repository import IReviewRepository
from cxt.shared.errors import AuthorizationError, ValidationError


@dataclass(frozen=True, slots=True)
class ReviewQueueItem:
    attempt: StudentAttempt


class ListReviewQueueUseCase:
    def __init__(self, *, attempt_repository: IAttemptRepository) -> None:
        self._attempts = attempt_repository

    def execute(self, *, caller_role: Role) -> list[StudentAttempt]:
        if not Role.can_review(caller_role):
            raise AuthorizationError("Only educators and admins can view the review queue.")
        return self._attempts.list_for_educator_review()


@dataclass(frozen=True, slots=True)
class ReviewAttemptRequest:
    attempt_id: str
    scores: dict[str, int]
    comment: str
    educator_id: str
    educator_role: Role


class ReviewAttemptUseCase:
    def __init__(
        self,
        *,
        review_repository: IReviewRepository,
        attempt_repository: IAttemptRepository,
    ) -> None:
        self._reviews = review_repository
        self._attempts = attempt_repository

    def execute(self, request: ReviewAttemptRequest) -> EducatorReview:
        if not Role.can_review(request.educator_role):
            raise AuthorizationError("Only educators and admins can submit reviews.")
        if not request.attempt_id:
            raise ValidationError("attempt_id is required.")
        if not isinstance(request.scores, dict) or not request.scores:
            raise ValidationError("scores must be a non-empty dict.")

        self._attempts.get(request.attempt_id)

        review = EducatorReview.new(
            id=str(uuid.uuid4()),
            attempt_id=request.attempt_id,
            educator_id=request.educator_id,
            scores=dict(request.scores),
            comment=request.comment or "",
        )
        self._reviews.add(review)
        return review


@dataclass(frozen=True, slots=True)
class ReviewsForAttemptResult:
    items: list[EducatorReview]


class ListReviewsForAttemptUseCase:
    def __init__(self, *, review_repository: IReviewRepository) -> None:
        self._reviews = review_repository

    def execute(self, *, attempt_id: str, caller_role: Role) -> list[EducatorReview]:
        if not Role.can_review(caller_role):
            raise AuthorizationError("Only educators and admins can view reviews.")
        return self._reviews.list_for_attempt(attempt_id)
