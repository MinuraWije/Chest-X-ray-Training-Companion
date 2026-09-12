"""Email value object: a validated, lowercased email address."""
from __future__ import annotations

import re

from cxt.shared.errors import ValidationError

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MAX_LEN = 254


class Email:
    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValidationError("Email must be a string.")
        normalized = value.strip().lower()
        if not normalized:
            raise ValidationError("Email must not be empty.")
        if len(normalized) > _MAX_LEN:
            raise ValidationError(f"Email must be at most {_MAX_LEN} characters.")
        if not _EMAIL_RE.match(normalized):
            raise ValidationError(f"Email is not a valid address: {value!r}.")
        self._value = normalized

    @property
    def value(self) -> str:
        return self._value

    def __str__(self) -> str:
        return self._value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Email):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)
