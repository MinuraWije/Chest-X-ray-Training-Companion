"""IAttemptRepository port: persistence for StudentAttempt entities."""
from __future__ import annotations

from typing import Protocol

from cxt.domain.entities.feedback import Feedback
from cxt.domain.entities.student_attempt import StudentAttempt


class IAttemptRepository(Protocol):
    def add(self, attempt: StudentAttempt) -> None: ...

    def get(self, attempt_id: str) -> StudentAttempt: ...

    def list_for_student(self, student_id: str) -> list[StudentAttempt]: ...

    def list_for_educator_review(self) -> list[StudentAttempt]: ...

    def attach_feedback(self, attempt_id: str, feedback: Feedback) -> None: ...

    def get_with_feedback(self, attempt_id: str) -> tuple[StudentAttempt, Feedback | None]: ...
