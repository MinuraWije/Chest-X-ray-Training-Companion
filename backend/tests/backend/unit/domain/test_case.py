from __future__ import annotations

import dataclasses

import pytest

from cxt.domain.entities.case import Case
from cxt.domain.entities.finding import Finding, FindingCategory
from cxt.domain.value_objects.finding_label import FindingLabel
from cxt.shared.errors import ValidationError


def _finding() -> Finding:
    return Finding(category=FindingCategory.PNEUMOTHORAX, label=FindingLabel("Pneumothorax"))


def test_case_creation_minimal() -> None:
    c = Case(id="c1", image_url="/images/c1.jpg", ground_truth_finding=_finding())
    assert c.id == "c1"
    assert c.split == "train"
    assert c.metadata == {}


def test_case_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        Case(id="", image_url="/x.jpg", ground_truth_finding=_finding())


def test_case_rejects_empty_image_url() -> None:
    with pytest.raises(ValidationError):
        Case(id="c1", image_url="", ground_truth_finding=_finding())


def test_case_rejects_non_finding_truth() -> None:
    with pytest.raises(ValidationError):
        Case(id="c1", image_url="/x.jpg", ground_truth_finding="pneumothorax")  # type: ignore[arg-type]


def test_case_is_frozen() -> None:
    c = Case(id="c1", image_url="/x.jpg", ground_truth_finding=_finding())
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        c.id = "c2"  # type: ignore[misc]
