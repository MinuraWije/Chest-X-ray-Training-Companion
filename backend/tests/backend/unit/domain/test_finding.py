from __future__ import annotations

import dataclasses

import pytest

from cxt.domain.entities.finding import Finding, FindingCategory
from cxt.domain.value_objects.finding_label import FindingLabel
from cxt.shared.errors import ValidationError


def test_finding_creation() -> None:
    f = Finding(category=FindingCategory.PNEUMOTHORAX, label=FindingLabel("Pneumothorax"))
    assert f.category is FindingCategory.PNEUMOTHORAX
    assert f.label.value == "Pneumothorax"


def test_finding_rejects_bad_category() -> None:
    with pytest.raises(ValidationError):
        Finding(category="pneumothorax", label=FindingLabel("Pneumothorax"))  # type: ignore[arg-type]


def test_finding_rejects_bad_label() -> None:
    with pytest.raises(ValidationError):
        Finding(category=FindingCategory.PNEUMOTHORAX, label="Pneumothorax")  # type: ignore[arg-type]


def test_finding_is_frozen() -> None:
    f = Finding(category=FindingCategory.CARDIOMEGALY, label=FindingLabel("Cardiomegaly"))
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        f.category = FindingCategory.FRACTURE  # type: ignore[misc]


def test_finding_category_has_all_six_plus_no_finding() -> None:
    expected = {
        FindingCategory.PNEUMOTHORAX,
        FindingCategory.PLEURAL_EFFUSION,
        FindingCategory.CARDIOMEGALY,
        FindingCategory.CONSOLIDATION,
        FindingCategory.NODULE_OR_MASS,
        FindingCategory.FRACTURE,
        FindingCategory.NO_FINDING,
    }
    assert set(FindingCategory) == expected
