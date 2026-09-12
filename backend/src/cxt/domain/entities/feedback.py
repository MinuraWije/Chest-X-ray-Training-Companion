"""Feedback entity: the structured, evidence-grounded output the system gives a student.

In Phase 1 `sources` is always empty. Phase 2 will populate it from RAG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cxt.domain.value_objects.confidence import Confidence
from cxt.domain.value_objects.finding_label import FindingLabel
from cxt.domain.value_objects.rationale import Rationale
from cxt.shared.errors import ValidationError


@dataclass(frozen=True, slots=True)
class FeedbackSection:
    """One labelled section of the feedback (likely finding, key signs, ...)."""

    heading: str
    body: str

    def __post_init__(self) -> None:
        if not isinstance(self.heading, str) or not self.heading.strip():
            raise ValidationError("FeedbackSection.heading must be a non-empty string.")
        if not isinstance(self.body, str) or not self.body.strip():
            raise ValidationError("FeedbackSection.body must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    """A passage retrieved from the RAG corpus. Phase 1 always empty."""

    text: str
    source: str
    score: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValidationError("RetrievedPassage.text must be a non-empty string.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValidationError("RetrievedPassage.source must be a non-empty string.")
        if not isinstance(self.score, int | float) or not 0.0 <= float(self.score) <= 1.0:
            raise ValidationError("RetrievedPassage.score must be in [0.0, 1.0].")


@dataclass(frozen=True, slots=True)
class DifferentialFinding:
    """An alternative diagnosis the model considered, with its confidence."""

    label: str
    confidence: Confidence

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValidationError("DifferentialFinding.label must be a non-empty string.")
        if not isinstance(self.confidence, Confidence):
            raise ValidationError("DifferentialFinding.confidence must be a Confidence.")


@dataclass(frozen=True, slots=True)
class Feedback:
    attempt_id: str
    likely_finding: FindingLabel
    key_signs: Rationale
    comparison: Rationale
    missed_clues: Rationale
    learning_point: Rationale
    confidence: Confidence
    # Defaulted so feedback_json rows written before these fields existed
    # still deserialize.
    key_observations: Rationale | None = None
    differentials: tuple[DifferentialFinding, ...] = field(default_factory=tuple)
    sources: tuple[RetrievedPassage, ...] = field(default_factory=tuple)
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.attempt_id:
            raise ValidationError("Feedback.attempt_id must be a non-empty string.")
        for name, value in (
            ("likely_finding", self.likely_finding),
            ("key_signs", self.key_signs),
            ("comparison", self.comparison),
            ("missed_clues", self.missed_clues),
            ("learning_point", self.learning_point),
            ("confidence", self.confidence),
        ):
            expected_type = {
                "likely_finding": FindingLabel,
                "key_signs": Rationale,
                "comparison": Rationale,
                "missed_clues": Rationale,
                "learning_point": Rationale,
                "confidence": Confidence,
            }[name]
            if not isinstance(value, expected_type):
                raise ValidationError(f"Feedback.{name} must be a {expected_type.__name__}.")
        if self.key_observations is not None and not isinstance(self.key_observations, Rationale):
            raise ValidationError("Feedback.key_observations must be a Rationale or None.")
        if not isinstance(self.differentials, tuple) or not all(
            isinstance(d, DifferentialFinding) for d in self.differentials
        ):
            raise ValidationError("Feedback.differentials must be a tuple of DifferentialFinding.")
        if not isinstance(self.sources, tuple) or not all(
            isinstance(s, RetrievedPassage) for s in self.sources
        ):
            raise ValidationError("Feedback.sources must be a tuple of RetrievedPassage.")
        if not isinstance(self.raw, dict):
            raise ValidationError("Feedback.raw must be a dict.")
