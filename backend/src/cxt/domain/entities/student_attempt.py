"""StudentAttempt entity: a student's free-text interpretation of a case."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from cxt.shared.errors import ValidationError


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class StudentAttempt:
    id: str
    case_id: str
    student_id: str
    free_text: str
    submitted_at: datetime

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValidationError("StudentAttempt.id must be a non-empty string.")
        if not self.case_id or not isinstance(self.case_id, str):
            raise ValidationError("StudentAttempt.case_id must be a non-empty string.")
        if not self.student_id or not isinstance(self.student_id, str):
            raise ValidationError("StudentAttempt.student_id must be a non-empty string.")
        if not isinstance(self.free_text, str) or not self.free_text.strip():
            raise ValidationError("StudentAttempt.free_text must be a non-empty string.")
        if not isinstance(self.submitted_at, datetime):
            raise ValidationError("StudentAttempt.submitted_at must be a datetime.")

    @classmethod
    def new(cls, *, id: str, case_id: str, student_id: str, free_text: str) -> StudentAttempt:
        return cls(
            id=id,
            case_id=case_id,
            student_id=student_id,
            free_text=free_text.strip(),
            submitted_at=_utcnow(),
        )
