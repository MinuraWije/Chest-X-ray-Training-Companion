from __future__ import annotations

import pytest

from cxt.domain.value_objects.rationale import Rationale
from cxt.shared.errors import ValidationError


def test_rationale_trims() -> None:
    assert Rationale("  Trachea is midline.  ").value == "Trachea is midline."


def test_rationale_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        Rationale("")


def test_rationale_rejects_too_long() -> None:
    with pytest.raises(ValidationError):
        Rationale("x" * 4001)


def test_rationale_rejects_non_string() -> None:
    with pytest.raises(ValidationError):
        Rationale(42)  # type: ignore[arg-type]
