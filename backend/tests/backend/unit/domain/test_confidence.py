from __future__ import annotations

import pytest

from cxt.domain.value_objects.confidence import Confidence
from cxt.shared.errors import ValidationError


def test_confidence_accepts_in_range() -> None:
    assert Confidence(0.0).value == 0.0
    assert Confidence(1.0).value == 1.0
    assert Confidence(0.5).value == 0.5


def test_confidence_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        Confidence(-0.01)
    with pytest.raises(ValidationError):
        Confidence(1.01)


def test_confidence_rejects_non_numeric() -> None:
    with pytest.raises(ValidationError):
        Confidence("0.5")  # type: ignore[arg-type]


def test_confidence_equality_with_float() -> None:
    assert Confidence(0.5) == 0.5
    assert float(Confidence(0.75)) == 0.75
