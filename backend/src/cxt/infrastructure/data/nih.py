"""NIH ChestX-ray8 / NIHCC 14-label <-> our six-finding mapping.

NIH labels are NLP-mined from radiology reports and are intentionally noisy.
This module is the single source of truth for translating between NIH's
14 disease labels and our six target findings. It also defines the rules for
"positive" and "negative" cases used in evaluation.

References:
    Wang et al., "ChestX-ray8: Hospital-Scale Chest X-ray Database and
    Benchmarks on Weakly-Supervised Classification and Localization of
    Common Thorax Diseases", CVPR 2017.
    NIHCC Data_Entry_2017.csv columns: Image Index, Finding Labels,
    Follow-up #, Patient ID, Patient Age, Patient Gender,
    View Position, OriginalImage[Width,Height] Pixel Spacing.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final

from cxt.domain.entities.finding import FindingCategory

NIH_FOURTEEN: Final[tuple[str, ...]] = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
)


NO_FINDING_LABEL: Final[str] = "No Finding"


class NihLabel:
    """A single NIH 14-label with helpers for our mapping."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("NihLabel must be a non-empty string.")
        self._value = value.strip()

    @property
    def value(self) -> str:
        return self._value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NihLabel):
            return self._value == other._value
        if isinstance(other, str):
            return self._value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return f"NihLabel({self._value!r})"


class NihMapping:
    """NIH 14-label -> our six findings.

    Each rule is a tuple of NIH labels that, when present, contribute to the
    FindingCategory on the right. A single NIH image can map to several of
    our findings.
    """

    PNEUMOTHORAX: Final[tuple[NihLabel, ...]] = (NihLabel("Pneumothorax"),)
    PLEURAL_EFFUSION: Final[tuple[NihLabel, ...]] = (NihLabel("Effusion"),)
    CARDIOMEGALY: Final[tuple[NihLabel, ...]] = (NihLabel("Cardiomegaly"),)
    CONSOLIDATION: Final[tuple[NihLabel, ...]] = (
        NihLabel("Consolidation"),
        NihLabel("Infiltration"),
        NihLabel("Pneumonia"),
    )
    NODULE_OR_MASS: Final[tuple[NihLabel, ...]] = (
        NihLabel("Nodule"),
        NihLabel("Mass"),
    )
    FRACTURE: Final[tuple[NihLabel, ...]] = (NihLabel("Fracture"),)

    RULES: Final[Mapping[FindingCategory, tuple[NihLabel, ...]]] = {
        FindingCategory.PNEUMOTHORAX: PNEUMOTHORAX,
        FindingCategory.PLEURAL_EFFUSION: PLEURAL_EFFUSION,
        FindingCategory.CARDIOMEGALY: CARDIOMEGALY,
        FindingCategory.CONSOLIDATION: CONSOLIDATION,
        FindingCategory.NODULE_OR_MASS: NODULE_OR_MASS,
        FindingCategory.FRACTURE: FRACTURE,
    }


def nih_labels_to_finding_categories(
    labels: Iterable[str],
) -> tuple[FindingCategory, ...]:
    """Map a NIH 'Finding Labels' cell (e.g. 'Cardiomegaly|Effusion') to
    our FindingCategory values. 'No Finding' yields (NO_FINDING,).

    Accepts either a list of single labels or an iterable of pipe-delimited
    strings; each cell is split on '|' before mapping.
    """
    flat: list[NihLabel] = []
    for cell in labels:
        if not cell:
            continue
        for part in cell.split("|"):
            stripped = part.strip()
            if stripped:
                flat.append(NihLabel(stripped))

    if not flat:
        return (FindingCategory.NO_FINDING,)

    if flat == [NihLabel(NO_FINDING_LABEL)]:
        return (FindingCategory.NO_FINDING,)

    matched: list[FindingCategory] = []
    for category, nih_labels in NihMapping.RULES.items():
        if any(label in nih_labels for label in flat):
            matched.append(category)
    if not matched:
        return (FindingCategory.NO_FINDING,)
    return tuple(matched)


def is_positive_for(labels: Iterable[str], target: FindingCategory) -> bool:
    """True iff a NIH 'Finding Labels' cell positively indicates `target`."""
    return target in nih_labels_to_finding_categories(labels)


def find_primary_category(labels: Iterable[str]) -> FindingCategory:
    """Pick a single primary FindingCategory for a NIH case.

    Used by the manifest builder when a case is positive for several
    findings; we deterministically pick the first by a fixed ordering so
    the split is reproducible across runs. The order is a clinical
    prioritisation: pneumothorax and pleural effusion first (don't miss
    these in the acute setting), then cardiomegaly, consolidation, and
    then fracture before nodule/mass (fracture is easier to confirm and
    more clinically actionable).
    """
    cats = nih_labels_to_finding_categories(labels)
    if cats == (FindingCategory.NO_FINDING,):
        return FindingCategory.NO_FINDING
    order = (
        FindingCategory.PNEUMOTHORAX,
        FindingCategory.PLEURAL_EFFUSION,
        FindingCategory.CARDIOMEGALY,
        FindingCategory.CONSOLIDATION,
        FindingCategory.FRACTURE,
        FindingCategory.NODULE_OR_MASS,
    )
    for c in order:
        if c in cats:
            return c
    return cats[0]
