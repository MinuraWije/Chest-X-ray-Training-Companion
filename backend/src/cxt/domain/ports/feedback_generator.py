"""IFeedbackGenerator port: assembles a Feedback entity for a student attempt.

Takes a case + attempt, runs the analyzer, and returns the structured
Feedback (with `sources` empty in Phase 1).
"""
from __future__ import annotations

from typing import Protocol

from cxt.domain.entities.case import Case
from cxt.domain.entities.feedback import Feedback
from cxt.domain.entities.student_attempt import StudentAttempt


class IFeedbackGenerator(Protocol):
    def generate(self, *, case: Case, attempt: StudentAttempt) -> Feedback: ...
