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
    """A passage retrieved from the teaching corpus.

    `source` is a human-readable citation ("WHO Manual of Diagnostic Imaging,
    p.42") and is deliberately **not** unique — a single document contributes
    many chunks, all carrying the same string. `id` is the chunk's stable
    corpus identifier and is what any keying, deduplication or citation
    enforcement must use. Both `id` and `url` default to empty so feedback
    rows persisted before Phase 2 still deserialize.
    """

    text: str
    source: str
    score: float = 1.0
    id: str = ""
    url: str = ""
    # Curated student-facing "Learn more" link. Empty when no verified link
    # exists for the chunk — the frontend hides the link rather than showing
    # a dead or guessed URL. Defaults keep pre-Phase-2 rows deserializing.
    read_more_url: str = ""
    read_more_label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValidationError("RetrievedPassage.text must be a non-empty string.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValidationError("RetrievedPassage.source must be a non-empty string.")
        if not isinstance(self.score, int | float) or not 0.0 <= float(self.score) <= 1.0:
            raise ValidationError("RetrievedPassage.score must be in [0.0, 1.0].")
        if not isinstance(self.id, str):
            raise ValidationError("RetrievedPassage.id must be a string.")
        if not isinstance(self.url, str):
            raise ValidationError("RetrievedPassage.url must be a string.")
        if not isinstance(self.read_more_url, str):
            raise ValidationError("RetrievedPassage.read_more_url must be a string.")
        if not isinstance(self.read_more_label, str):
            raise ValidationError("RetrievedPassage.read_more_label must be a string.")
        if self.read_more_url and not self.read_more_url.startswith("https://"):
            raise ValidationError(
                "RetrievedPassage.read_more_url must be an https:// URL."
            )
        if self.read_more_url and not self.read_more_label.strip():
            raise ValidationError(
                "RetrievedPassage.read_more_label is required when read_more_url is set."
            )


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
