"""Rationale value object: a non-empty, length-bounded textual explanation."""
from __future__ import annotations

from cxt.shared.errors import ValidationError

_MIN_LEN = 1
_MAX_LEN = 4000


class Rationale:
    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValidationError("Rationale must be a string.")
        normalized = value.strip()
        if len(normalized) < _MIN_LEN:
            raise ValidationError("Rationale must not be empty.")
        if len(normalized) > _MAX_LEN:
            raise ValidationError(f"Rationale must be at most {_MAX_LEN} characters.")
        self._value = normalized

    @property
    def value(self) -> str:
        return self._value

    def __str__(self) -> str:
        return self._value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Rationale):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)
