from __future__ import annotations

import pytest

from cxt.domain.entities.role import Role
from cxt.domain.entities.user import User
from cxt.domain.value_objects.email import Email
from cxt.shared.errors import ValidationError


def test_user_new_sets_timestamps_and_keeps_role() -> None:
    u = User.new(
        id="u1",
        email=Email("a@b.com"),
        password_hash="$argon2id$...",
        role=Role.STUDENT,
    )
    assert u.id == "u1"
    assert u.email.value == "a@b.com"
    assert u.password_hash == "$argon2id$..."
    assert u.role is Role.STUDENT
    assert u.created_at is not None


def test_user_can_review_predicate() -> None:
    assert User.new(id="1", email=Email("a@b.com"), password_hash="x", role=Role.EDUCATOR).can_review() is True
    assert User.new(id="2", email=Email("a@b.com"), password_hash="x", role=Role.ADMIN).can_review() is True
    assert User.new(id="3", email=Email("a@b.com"), password_hash="x", role=Role.STUDENT).can_review() is False


def test_user_rejects_bad_email() -> None:
    with pytest.raises(ValidationError):
        User.new(id="1", email="not-an-email", password_hash="x", role=Role.STUDENT)  # type: ignore[arg-type]


def test_user_rejects_empty_password_hash() -> None:
    with pytest.raises(ValidationError):
        User.new(id="1", email=Email("a@b.com"), password_hash="", role=Role.STUDENT)


def test_user_rejects_bad_role() -> None:
    with pytest.raises(ValidationError):
        User.new(id="1", email=Email("a@b.com"), password_hash="x", role="STUDENT")  # type: ignore[arg-type]
