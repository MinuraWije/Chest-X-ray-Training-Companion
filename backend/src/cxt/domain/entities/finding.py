"""FindingCategory enum and Finding entity.

The six target findings (from the project proposal) plus an explicit
NO_FINDING category for negative / clean cases in the curated dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cxt.domain.value_objects.finding_label import FindingLabel
from cxt.shared.errors import ValidationError


class FindingCategory(str, Enum):
    PNEUMOTHORAX = "pneumothorax"
    PLEURAL_EFFUSION = "pleural_effusion"
    CARDIOMEGALY = "cardiomegaly"
    CONSOLIDATION = "consolidation"
    NODULE_OR_MASS = "nodule_or_mass"
    FRACTURE = "fracture"
    NO_FINDING = "no_finding"


@dataclass(frozen=True, slots=True)
class Finding:
    category: FindingCategory
    label: FindingLabel

    def __post_init__(self) -> None:
        if not isinstance(self.category, FindingCategory):
            raise ValidationError(f"Finding.category must be a FindingCategory, got {type(self.category).__name__}.")
        if not isinstance(self.label, FindingLabel):
            raise ValidationError(f"Finding.label must be a FindingLabel, got {type(self.label).__name__}.")
