from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cxt.domain.entities.role import Role
from cxt.domain.entities.user import User
from cxt.domain.value_objects.email import Email
from cxt.infrastructure.auth.jwt_auth_service import JwtAuthService
from cxt.infrastructure.config.settings import MedGemmaRuntime, Settings, get_settings
from cxt.infrastructure.data.database import (
    build_engine,
    build_session_factory,
    init_database,
)
from cxt.infrastructure.data.sqlite_user_repository import SqliteUserRepository
from cxt.interfaces.api.app import create_app
from cxt.interfaces.api.dependencies import reset_dependency_caches_for_tests


@pytest.fixture()
def database_path(tmp_path: Path) -> Path:
    return tmp_path / "cxt.db"


@pytest.fixture()
def settings(database_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite+pysqlite:///{database_path}",
        medgemma_runtime=MedGemmaRuntime.DRY_RUN,
        auth_jwt_secret="x" * 32,
    )


@pytest.fixture()
def client(settings: Settings) -> Iterator[TestClient]:
    reset_dependency_caches_for_tests()
    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()
    reset_dependency_caches_for_tests()


def _seed_admin(settings: Settings, email: str = "admin@example.com") -> tuple[str, str]:
    """Bootstrap the first admin (mirrors scripts/seed_admin.py).

    Returns (admin_id, access_token). The auth service uses the same secret
    as the app, so the token is valid for the test client.
    """
    engine = build_engine(settings)
    init_database(engine)
    factory = build_session_factory(engine)
    users = SqliteUserRepository(factory)
    auth = JwtAuthService(
        secret=settings.auth_jwt_secret,
        algorithm=settings.auth_jwt_algorithm,
        ttl_seconds=settings.auth_access_token_ttl_seconds,
        issuer=settings.auth_jwt_issuer,
        audience=settings.auth_jwt_audience,
    )
    admin = User.new(
        id=str(uuid.uuid4()),
        email=Email(email),
        password_hash=auth.hash_password("admin-pw-1"),
        role=Role.ADMIN,
    )
    users.add(admin)
    token, _ = auth.issue_token(user_id=admin.id, role=admin.role)
    return admin.id, token


class TestRegister:
    def test_student_open(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "alice@example.com", "password": "long-pw-1", "role": "STUDENT"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["accessToken"]
        assert body["user"]["email"] == "alice@example.com"
        assert body["user"]["role"] == "STUDENT"

    def test_educator_without_invite_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "edu@example.com", "password": "long-pw-1", "role": "EDUCATOR"},
        )
        assert response.status_code == 403

    def test_educator_with_admin_invite_succeeds(
        self, client: TestClient, settings: Settings
    ) -> None:
        admin_id, _ = _seed_admin(settings, "admin1@example.com")
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "edu@example.com",
                "password": "long-pw-1",
                "role": "EDUCATOR",
                "inviteAdminId": admin_id,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["user"]["role"] == "EDUCATOR"

    def test_short_password_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "x@example.com", "password": "short", "role": "STUDENT"},
        )
        assert response.status_code == 422

    def test_duplicate_email_rejected(self, client: TestClient) -> None:
        first = client.post(
            "/api/v1/auth/register",
            json={"email": "dup@example.com", "password": "long-pw-1", "role": "STUDENT"},
        )
        assert first.status_code == 200
        second = client.post(
            "/api/v1/auth/register",
            json={"email": "dup@example.com", "password": "long-pw-1", "role": "STUDENT"},
        )
        assert second.status_code == 409


class TestLogin:
    def test_happy_path(self, client: TestClient) -> None:
        client.post(
            "/api/v1/auth/register",
            json={"email": "bob@example.com", "password": "long-pw-1", "role": "STUDENT"},
        )
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "bob@example.com", "password": "long-pw-1"},
        )
        assert response.status_code == 200
        assert response.json()["accessToken"]

    def test_wrong_password_returns_401(self, client: TestClient) -> None:
        client.post(
            "/api/v1/auth/register",
            json={"email": "c@example.com", "password": "long-pw-1", "role": "STUDENT"},
        )
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "c@example.com", "password": "wrong-pw-12"},
        )
        assert response.status_code == 401

    def test_unknown_email_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "long-pw-1"},
        )
        assert response.status_code == 401


class TestMe:
    def test_returns_caller(self, client: TestClient) -> None:
        reg = client.post(
            "/api/v1/auth/register",
            json={"email": "me@example.com", "password": "long-pw-1", "role": "STUDENT"},
        )
        token = reg.json()["accessToken"]
        response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        body = response.json()
        assert body["email"] == "me@example.com"
        assert body["role"] == "STUDENT"

    def test_no_token_returns_401(self, client: TestClient) -> None:
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401

    def test_garbage_token_returns_401(self, client: TestClient) -> None:
        response = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
        assert response.status_code == 401
