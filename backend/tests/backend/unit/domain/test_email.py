from __future__ import annotations

import pytest

from cxt.domain.value_objects.email import Email
from cxt.shared.errors import ValidationError


def test_email_lowercases_and_trims() -> None:
    e = Email("  Foo@Example.COM  ")
    assert e.value == "foo@example.com"
    assert str(e) == "foo@example.com"


def test_email_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        Email("")
    with pytest.raises(ValidationError):
        Email("   ")


def test_email_rejects_invalid_shape() -> None:
    for bad in ["no-at-sign", "a@", "@b.com", "a@b", "a b@c.com"]:
        with pytest.raises(ValidationError):
            Email(bad)


def test_email_rejects_too_long() -> None:
    long_local = "a" * 250
    with pytest.raises(ValidationError):
        Email(f"{long_local}@x.io")


def test_email_rejects_non_string() -> None:
    with pytest.raises(ValidationError):
        Email(123)  # type: ignore[arg-type]


def test_email_equality_and_hash() -> None:
    a = Email("foo@bar.com")
    b = Email("FOO@bar.com")
    assert a == b
    assert hash(a) == hash(b)
