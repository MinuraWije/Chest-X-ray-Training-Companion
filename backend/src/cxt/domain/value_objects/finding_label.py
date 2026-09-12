"""FindingLabel value object: a non-empty, trimmed label for a chest X-ray finding."""
from __future__ import annotations

from cxt.shared.errors import ValidationError

_MAX_LEN = 200


class FindingLabel:
    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValidationError("FindingLabel must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValidationError("FindingLabel must not be empty.")
        if len(normalized) > _MAX_LEN:
            raise ValidationError(f"FindingLabel must be at most {_MAX_LEN} characters.")
        self._value = normalized

    @property
    def value(self) -> str:
        return self._value

    def __str__(self) -> str:
        return self._value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FindingLabel):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)
