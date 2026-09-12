"""EducatorReview entity: an educator's rubric-based scoring of a student attempt."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from cxt.shared.errors import ValidationError

_VALID_CRITERIA = frozenset(
    {
        "finding_recognition",
        "retrieval_relevance",
        "evidence_grounding",
        "clarity",
        "usefulness_for_learning",
        "safety_trustworthiness",
    }
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class EducatorReview:
    id: str
    attempt_id: str
    educator_id: str
    scores: dict[str, int]
    comment: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValidationError("EducatorReview.id must be a non-empty string.")
        if not self.attempt_id or not isinstance(self.attempt_id, str):
            raise ValidationError("EducatorReview.attempt_id must be a non-empty string.")
        if not self.educator_id or not isinstance(self.educator_id, str):
            raise ValidationError("EducatorReview.educator_id must be a non-empty string.")
        if not isinstance(self.scores, dict) or not self.scores:
            raise ValidationError("EducatorReview.scores must be a non-empty dict.")
        unknown = set(self.scores) - _VALID_CRITERIA
        if unknown:
            raise ValidationError(f"EducatorReview.scores has unknown criteria: {sorted(unknown)}.")
        for name, value in self.scores.items():
            if not isinstance(value, int) or not 1 <= value <= 5:
                raise ValidationError(f"EducatorReview.scores[{name!r}] must be an int in 1..5, got {value!r}.")
        if not isinstance(self.comment, str):
            raise ValidationError("EducatorReview.comment must be a string.")
        if not isinstance(self.created_at, datetime):
            raise ValidationError("EducatorReview.created_at must be a datetime.")
        if not isinstance(self.metadata, dict):
            raise ValidationError("EducatorReview.metadata must be a dict.")

    @classmethod
    def new(
        cls,
        *,
        id: str,
        attempt_id: str,
        educator_id: str,
        scores: dict[str, int],
        comment: str,
    ) -> EducatorReview:
        return cls(
            id=id,
            attempt_id=attempt_id,
            educator_id=educator_id,
            scores=dict(scores),
            comment=comment.strip(),
            created_at=_utcnow(),
        )

    @property
    def average_score(self) -> float:
        return sum(self.scores.values()) / len(self.scores)
