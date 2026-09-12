from __future__ import annotations

import pytest

from cxt.domain.value_objects.finding_label import FindingLabel
from cxt.shared.errors import ValidationError


def test_finding_label_trims() -> None:
    assert FindingLabel("  Pneumothorax  ").value == "Pneumothorax"


def test_finding_label_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        FindingLabel("")
    with pytest.raises(ValidationError):
        FindingLabel("   ")


def test_finding_label_rejects_too_long() -> None:
    with pytest.raises(ValidationError):
        FindingLabel("x" * 201)


def test_finding_label_equality() -> None:
    assert FindingLabel("Effusion") == FindingLabel("Effusion")
    assert FindingLabel("A") != FindingLabel("B")
