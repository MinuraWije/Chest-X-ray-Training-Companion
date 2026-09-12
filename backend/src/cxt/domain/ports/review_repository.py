"""IReviewRepository port: persistence for EducatorReview entities."""
from __future__ import annotations

from typing import Protocol

from cxt.domain.entities.review import EducatorReview


class IReviewRepository(Protocol):
    def add(self, review: EducatorReview) -> None: ...

    def get(self, review_id: str) -> EducatorReview: ...

    def list_for_attempt(self, attempt_id: str) -> list[EducatorReview]: ...
