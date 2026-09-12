"""Tests for SqliteUserRepository."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from cxt.domain.entities.role import Role
from cxt.domain.entities.user import User
from cxt.domain.value_objects.email import Email
from cxt.infrastructure.config.settings import Settings
from cxt.infrastructure.data.database import (
    build_engine,
    build_session_factory,
    init_database,
)
from cxt.infrastructure.data.sqlite_user_repository import SqliteUserRepository
from cxt.shared.errors import ConflictError, NotFoundError


@pytest.fixture()
def users(tmp_path: Path) -> Iterator[SqliteUserRepository]:
    settings = Settings(database_url=f"sqlite+pysqlite:///{tmp_path / 'cxt.db'}")
    engine = build_engine(settings)
    init_database(engine)
    factory = build_session_factory(engine)
    yield SqliteUserRepository(factory)


def _new_user(id_: str, email: str = "a@b.com", role: Role = Role.STUDENT) -> User:
    return User.new(id=id_, email=Email(email), password_hash="h", role=role)


class TestAddAndGet:
    def test_round_trip(self, users: SqliteUserRepository) -> None:
        users.add(_new_user("u-1"))
        u = users.get("u-1")
        assert u.id == "u-1"
        assert u.email.value == "a@b.com"
        assert u.role is Role.STUDENT

    def test_get_unknown_raises(self, users: SqliteUserRepository) -> None:
        with pytest.raises(NotFoundError):
            users.get("nope")


class TestFindByEmail:
    def test_match(self, users: SqliteUserRepository) -> None:
        users.add(_new_user("u-1", "match@example.com"))
        u = users.find_by_email(Email("match@example.com"))
        assert u is not None
        assert u.id == "u-1"

    def test_case_insensitive_lookup(self, users: SqliteUserRepository) -> None:
        users.add(_new_user("u-1", "MiX@Example.com"))
        u = users.find_by_email(Email("mix@example.com"))
        assert u is not None
        assert u.id == "u-1"

    def test_no_match(self, users: SqliteUserRepository) -> None:
        assert users.find_by_email(Email("nope@example.com")) is None


class TestConflict:
    def test_duplicate_email_raises(self, users: SqliteUserRepository) -> None:
        users.add(_new_user("u-1", "dup@example.com"))
        with pytest.raises(ConflictError):
            users.try_add(_new_user("u-2", "dup@example.com"))


class TestList:
    def test_returns_all(self, users: SqliteUserRepository) -> None:
        users.add(_new_user("u-1", "a@x.com", Role.STUDENT))
        users.add(_new_user("u-2", "b@x.com", Role.EDUCATOR))
        items = users.list()
        assert {u.id for u in items} == {"u-1", "u-2"}
