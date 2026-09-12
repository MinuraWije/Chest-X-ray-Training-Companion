from __future__ import annotations

import pytest

from cxt.domain.entities.role import Role


def test_role_values() -> None:
    assert Role.STUDENT.value == "STUDENT"
    assert Role.EDUCATOR.value == "EDUCATOR"
    assert Role.ADMIN.value == "ADMIN"


def test_role_can_review() -> None:
    assert Role.can_review(Role.EDUCATOR) is True
    assert Role.can_review(Role.ADMIN) is True
    assert Role.can_review(Role.STUDENT) is False


def test_role_can_submit_attempt() -> None:
    assert Role.can_submit_attempt(Role.STUDENT) is True
    assert Role.can_submit_attempt(Role.EDUCATOR) is False
    assert Role.can_submit_attempt(Role.ADMIN) is False


def test_role_is_string_enum() -> None:
    assert Role("STUDENT") is Role.STUDENT
    with pytest.raises(ValueError):
        Role("nope")  # type: ignore[call-arg]
