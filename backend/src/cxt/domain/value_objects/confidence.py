"""Confidence value object: a model self-reported confidence in [0.0, 1.0]."""
from __future__ import annotations

from cxt.shared.errors import ValidationError


class Confidence:
    __slots__ = ("_value",)

    def __init__(self, value: float) -> None:
        if not isinstance(value, int | float):
            raise ValidationError("Confidence must be a number.")
        as_float = float(value)
        if as_float < 0.0 or as_float > 1.0:
            raise ValidationError(f"Confidence must be in [0.0, 1.0], got {as_float}.")
        self._value = as_float

    @property
    def value(self) -> float:
        return self._value

    def __float__(self) -> float:
        return self._value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Confidence):
            return self._value == other._value
        if isinstance(other, int | float):
            return self._value == float(other)
        return NotImplemented
