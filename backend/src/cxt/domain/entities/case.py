"""Case entity: a chest X-ray case shown to a student.

The image is identified by `image_url`; the URL is resolved by the data layer
(typically `IMAGES_DIR/{case_id}.jpg` served by FastAPI's StaticFiles).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cxt.domain.entities.finding import Finding
from cxt.shared.errors import ValidationError


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    image_url: str
    ground_truth_finding: Finding
    split: str = "train"
    # Other findings the case's labels confirm, beyond the primary. A third of
    # the test split carries more than one of our six categories, and
    # `find_primary_category` keeps only the highest-priority one — so without
    # this the case record asserts a single finding on a film that has two, and
    # a student who correctly spots the other is marked against an answer key
    # that does not contain it. Defaulted to empty: a single-finding case and a
    # hand-built one both behave exactly as before.
    secondary_findings: tuple[Finding, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def patient_id(self) -> str:
        """The patient this film belongs to, or "" when unknown.

        NIH names every image `<patient>_<follow-up>.png`, so the patient is
        recoverable from the index without a separate column. It matters
        because the curated subset is 300 images from only 86 patients: one
        patient contributes 47. Sampling that ignores this draws the same chest
        repeatedly and reports it as coverage.
        """
        index = self.metadata.get("image_index", "")
        if not isinstance(index, str) or "_" not in index:
            return ""
        return index.split("_", 1)[0]

    @property
    def confirmed_categories(self) -> tuple[str, ...]:
        """Every category this case's labels confirm, primary first."""
        seen = [self.ground_truth_finding.category.value]
        for extra in self.secondary_findings:
            if extra.category.value not in seen:
                seen.append(extra.category.value)
        return tuple(seen)

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValidationError("Case.id must be a non-empty string.")
        if not self.image_url or not isinstance(self.image_url, str):
            raise ValidationError("Case.image_url must be a non-empty string.")
        if not isinstance(self.ground_truth_finding, Finding):
            raise ValidationError(
                f"Case.ground_truth_finding must be a Finding, got {type(self.ground_truth_finding).__name__}."
            )
        if not isinstance(self.split, str) or not self.split:
            raise ValidationError("Case.split must be a non-empty string.")
        if not isinstance(self.secondary_findings, tuple):
            raise ValidationError("Case.secondary_findings must be a tuple of Finding.")
        for extra in self.secondary_findings:
            if not isinstance(extra, Finding):
                raise ValidationError(
                    f"Case.secondary_findings must hold Finding, got {type(extra).__name__}."
                )
            if extra.category == self.ground_truth_finding.category:
                # Listing the primary twice would have the prompt tell the model
                # the same finding is both the confirmed answer and an
                # additional one.
                raise ValidationError(
                    f"Case.secondary_findings repeats the primary category "
                    f"{extra.category.value!r}."
                )
        if not isinstance(self.metadata, dict):
            raise ValidationError("Case.metadata must be a dict.")
