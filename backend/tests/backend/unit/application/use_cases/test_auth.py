"""Tests for the auth use cases (register, login, get-me)."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from cxt.application.use_cases.auth import (
    GetMeUseCase,
    LoginRequest,
    LoginUseCase,
    RegisterRequest,
    RegisterUseCase,
)
from cxt.domain.entities.role import Role
from cxt.domain.entities.user import User
from cxt.domain.value_objects.email import Email
from cxt.infrastructure.auth.jwt_auth_service import JwtAuthService
from cxt.infrastructure.config.settings import Settings
from cxt.infrastructure.data.database import (
    build_engine,
    build_session_factory,
    init_database,
)
from cxt.infrastructure.data.sqlite_user_repository import SqliteUserRepository
from cxt.shared.errors import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    ValidationError,
)


@pytest.fixture()
def auth_service() -> JwtAuthService:
    return JwtAuthService(
        secret="x" * 32,
        algorithm="HS256",
        ttl_seconds=60,
        issuer="cxt-companion",
        audience="cxt-companion",
    )


@pytest.fixture()
def users(tmp_path: Path) -> Iterator[SqliteUserRepository]:
    settings = Settings(database_url=f"sqlite+pysqlite:///{tmp_path / 'cxt.db'}")
    engine = build_engine(settings)
    init_database(engine)
    factory = build_session_factory(engine)
    yield SqliteUserRepository(factory)


class TestRegisterUseCase:
    def test_open_student_registration(self, users: SqliteUserRepository, auth_service: JwtAuthService) -> None:
        use_case = RegisterUseCase(user_repository=users, auth_service=auth_service)
        result = use_case.execute(
            RegisterRequest(email="student@example.com", password="long-pw-1", role=Role.STUDENT)
        )
        assert result.user.email.value == "student@example.com"
        assert result.user.role is Role.STUDENT
        assert result.access_token
        assert result.claims.sub == result.user.id
        assert result.claims.role is Role.STUDENT

    def test_educator_creation_requires_admin_invite(
        self, users: SqliteUserRepository, auth_service: JwtAuthService
    ) -> None:
        use_case = RegisterUseCase(user_repository=users, auth_service=auth_service)
        with pytest.raises(AuthorizationError):
            use_case.execute(
                RegisterRequest(email="edu@example.com", password="long-pw-1", role=Role.EDUCATOR)
            )

    def test_educator_creation_with_non_admin_inviter(
        self, users: SqliteUserRepository, auth_service: JwtAuthService
    ) -> None:
        # Seed a student, then try to invite an educator as that student.
        student = User.new(
            id="u-s",
            email=Email("s@example.com"),
            password_hash=auth_service.hash_password("long-pw-1"),
            role=Role.STUDENT,
        )
        users.add(student)
        use_case = RegisterUseCase(user_repository=users, auth_service=auth_service)
        with pytest.raises(AuthorizationError):
            use_case.execute(
                RegisterRequest(
                    email="edu@example.com",
                    password="long-pw-1",
                    role=Role.EDUCATOR,
                    invite_admin_id=student.id,
                )
            )

    def test_educator_creation_with_admin_invite(
        self, users: SqliteUserRepository, auth_service: JwtAuthService
    ) -> None:
        admin = User.new(
            id="u-a",
            email=Email("a@example.com"),
            password_hash=auth_service.hash_password("long-pw-1"),
            role=Role.ADMIN,
        )
        users.add(admin)
        use_case = RegisterUseCase(user_repository=users, auth_service=auth_service)
        result = use_case.execute(
            RegisterRequest(
                email="edu@example.com",
                password="long-pw-1",
                role=Role.EDUCATOR,
                invite_admin_id=admin.id,
            )
        )
        assert result.user.role is Role.EDUCATOR

    def test_duplicate_email_rejected(
        self, users: SqliteUserRepository, auth_service: JwtAuthService
    ) -> None:
        use_case = RegisterUseCase(user_repository=users, auth_service=auth_service)
        use_case.execute(
            RegisterRequest(email="dup@example.com", password="long-pw-1", role=Role.STUDENT)
        )
        with pytest.raises(ConflictError):
            use_case.execute(
                RegisterRequest(email="dup@example.com", password="long-pw-1", role=Role.STUDENT)
            )

    def test_short_password_rejected(
        self, users: SqliteUserRepository, auth_service: JwtAuthService
    ) -> None:
        use_case = RegisterUseCase(user_repository=users, auth_service=auth_service)
        with pytest.raises(ValidationError):
            use_case.execute(
                RegisterRequest(email="x@example.com", password="short", role=Role.STUDENT)
            )


class TestLoginUseCase:
    def _seed(self, users: SqliteUserRepository, auth_service: JwtAuthService) -> User:
        user = User.new(
            id="u1",
            email=Email("login@example.com"),
            password_hash=auth_service.hash_password("long-pw-1"),
            role=Role.STUDENT,
        )
        users.add(user)
        return user

    def test_happy_path(self, users: SqliteUserRepository, auth_service: JwtAuthService) -> None:
        self._seed(users, auth_service)
        use_case = LoginUseCase(user_repository=users, auth_service=auth_service)
        result = use_case.execute(LoginRequest(email="login@example.com", password="long-pw-1"))
        assert result.user.id == "u1"
        assert result.access_token

    def test_wrong_password(self, users: SqliteUserRepository, auth_service: JwtAuthService) -> None:
        self._seed(users, auth_service)
        use_case = LoginUseCase(user_repository=users, auth_service=auth_service)
        with pytest.raises(AuthenticationError):
            use_case.execute(LoginRequest(email="login@example.com", password="wrong-wrong-1"))

    def test_unknown_email(self, users: SqliteUserRepository, auth_service: JwtAuthService) -> None:
        use_case = LoginUseCase(user_repository=users, auth_service=auth_service)
        with pytest.raises(AuthenticationError):
            use_case.execute(LoginRequest(email="nobody@example.com", password="long-pw-1"))

    def test_short_password_rejected(
        self, users: SqliteUserRepository, auth_service: JwtAuthService
    ) -> None:
        use_case = LoginUseCase(user_repository=users, auth_service=auth_service)
        with pytest.raises(ValidationError):
            use_case.execute(LoginRequest(email="login@example.com", password="x"))


class TestGetMeUseCase:
    def test_returns_user(self, users: SqliteUserRepository) -> None:
        user = User.new(
            id="u1",
            email=Email("me@example.com"),
            password_hash="h",
            role=Role.STUDENT,
        )
        users.add(user)
        use_case = GetMeUseCase(user_repository=users)
        assert use_case.execute(user_id="u1").email.value == "me@example.com"

    def test_empty_id_rejected(self, users: SqliteUserRepository) -> None:
        use_case = GetMeUseCase(user_repository=users)
        with pytest.raises(AuthenticationError):
            use_case.execute(user_id="")
