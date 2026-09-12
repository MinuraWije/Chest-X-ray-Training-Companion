from __future__ import annotations

import csv
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
from cxt.interfaces.api.dependencies import (
    reset_dependency_caches_for_tests,
)


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Path:
    rows = [
        {
            "case_id": "case_aaaa1111",
            "image_index": "00000001_000",
            "image_path": "images/00000001_000.png",
            "finding": "pneumothorax",
            "label_source": "stub",
            "split": "train",
            "nih_labels": "Pneumothorax",
        },
        {
            "case_id": "case_bbbb2222",
            "image_index": "00000002_000",
            "image_path": "images/00000002_000.png",
            "finding": "no_finding",
            "label_source": "stub",
            "split": "train",
            "nih_labels": "No Finding",
        },
    ]
    path = tmp_path / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


@pytest.fixture()
def database_url(tmp_path: Path) -> Path:
    path = tmp_path / "cxt.db"
    return path


@pytest.fixture()
def settings(manifest_path: Path, database_url: Path) -> Settings:
    return Settings(
        manifest_path=manifest_path,
        database_url=f"sqlite+pysqlite:///{database_url}",
        medgemma_runtime=MedGemmaRuntime.DRY_RUN,
        auth_jwt_secret="x" * 32,
    )


@pytest.fixture()
def auth_service(settings: Settings) -> JwtAuthService:
    return JwtAuthService(
        secret=settings.auth_jwt_secret,
        algorithm=settings.auth_jwt_algorithm,
        ttl_seconds=settings.auth_access_token_ttl_seconds,
        issuer=settings.auth_jwt_issuer,
        audience=settings.auth_jwt_audience,
    )


@pytest.fixture()
def student_user(auth_service: JwtAuthService, settings: Settings) -> User:
    engine = build_engine(settings)
    init_database(engine)
    factory = build_session_factory(engine)
    users = SqliteUserRepository(factory)
    user = User.new(
        id="u-student-1",
        email=Email("student@example.com"),
        password_hash=auth_service.hash_password("student-pw-1"),
        role=Role.STUDENT,
    )
    users.add(user)
    return user


@pytest.fixture()
def student_token(student_user: User, auth_service: JwtAuthService) -> str:
    token, _ = auth_service.issue_token(user_id=student_user.id, role=student_user.role)
    return token


@pytest.fixture()
def student_headers(student_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {student_token}"}


@pytest.fixture()
def client(
    settings: Settings,
    student_user: User,
    student_token: str,
) -> Iterator[TestClient]:
    reset_dependency_caches_for_tests()
    get_settings.cache_clear()

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()
    reset_dependency_caches_for_tests()


# Long enough to clear the route's 20-character floor.
_VALID_FREE_TEXT = "Left apical lucency, no lung markings."


class TestSubmitAttemptRoute:
    def test_happy_path(self, client: TestClient, student_headers: dict[str, str]) -> None:
        response = client.post(
            "/api/v1/attempts",
            json={"caseId": "case_aaaa1111", "freeText": "Small left apical pneumothorax."},
            headers=student_headers,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["caseId"] == "case_aaaa1111"
        assert body["studentId"] == "u-student-1"
        assert body["feedback"]["primaryFinding"]["category"]
        assert 0.0 <= body["feedback"]["primaryFinding"]["confidence"] <= 1.0

    def test_unauthenticated_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/attempts",
            json={"caseId": "case_aaaa1111", "freeText": _VALID_FREE_TEXT},
        )
        assert response.status_code == 401

    def test_unknown_case_returns_404(self, client: TestClient, student_headers: dict[str, str]) -> None:
        response = client.post(
            "/api/v1/attempts",
            json={"caseId": "case_doesnotexist", "freeText": _VALID_FREE_TEXT},
            headers=student_headers,
        )
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_missing_case_id_is_422(self, client: TestClient, student_headers: dict[str, str]) -> None:
        response = client.post(
            "/api/v1/attempts",
            json={"freeText": _VALID_FREE_TEXT},
            headers=student_headers,
        )
        assert response.status_code == 422

    def test_empty_free_text_is_422(self, client: TestClient, student_headers: dict[str, str]) -> None:
        response = client.post(
            "/api/v1/attempts",
            json={"caseId": "case_aaaa1111", "freeText": ""},
            headers=student_headers,
        )
        assert response.status_code == 422

    def test_attempt_persists_and_can_be_fetched(
        self, client: TestClient, student_headers: dict[str, str]
    ) -> None:
        post = client.post(
            "/api/v1/attempts",
            json={"caseId": "case_aaaa1111", "freeText": _VALID_FREE_TEXT},
            headers=student_headers,
        )
        assert post.status_code == 200
        att_id = post.json()["id"]
        get = client.get(f"/api/v1/attempts/{att_id}", headers=student_headers)
        assert get.status_code == 200
        assert get.json()["id"] == att_id
        assert get.json()["feedback"] is not None

    def test_get_unknown_attempt_returns_404(
        self, client: TestClient, student_headers: dict[str, str]
    ) -> None:
        response = client.get("/api/v1/attempts/not-a-real-id", headers=student_headers)
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_student_cannot_view_other_students_attempt(
        self, client: TestClient, student_headers: dict[str, str], settings: Settings
    ) -> None:
        post = client.post(
            "/api/v1/attempts",
            json={"caseId": "case_aaaa1111", "freeText": _VALID_FREE_TEXT},
            headers=student_headers,
        )
        att_id = post.json()["id"]
        other_token, _ = JwtAuthService(
            secret=settings.auth_jwt_secret,
            algorithm=settings.auth_jwt_algorithm,
            ttl_seconds=settings.auth_access_token_ttl_seconds,
            issuer=settings.auth_jwt_issuer,
            audience=settings.auth_jwt_audience,
        ).issue_token(user_id="u-other", role=Role.STUDENT)
        get = client.get(
            f"/api/v1/attempts/{att_id}",
            headers={"Authorization": f"Bearer {other_token}"},
        )
        assert get.status_code == 404


class TestInventoryIncludesAttempts:
    def test_inventory_lists_attempts(self, client: TestClient) -> None:
        body = client.get("/api/v1/").json()
        paths = {(r["method"], r["path"]) for r in body["routes"]}
        assert ("POST", "/api/v1/attempts") in paths
        assert ("GET", "/api/v1/attempts/{attempt_id}") in paths
        assert body["phase"]


class TestMissingManifest:
    def test_attempts_503_when_manifest_missing(self, tmp_path: Path, database_url: Path) -> None:
        reset_dependency_caches_for_tests()
        get_settings.cache_clear()
        settings = Settings(
            manifest_path=tmp_path / "missing.csv",
            database_url=f"sqlite+pysqlite:///{database_url}",
            medgemma_runtime=MedGemmaRuntime.DRY_RUN,
            auth_jwt_secret="x" * 32,
        )
        auth = JwtAuthService(
            secret=settings.auth_jwt_secret,
            algorithm=settings.auth_jwt_algorithm,
            ttl_seconds=settings.auth_access_token_ttl_seconds,
            issuer=settings.auth_jwt_issuer,
            audience=settings.auth_jwt_audience,
        )
        engine = build_engine(settings)
        init_database(engine)
        factory = build_session_factory(engine)
        users = SqliteUserRepository(factory)
        users.add(
            User.new(
                id="u-student-2",
                email=Email("s2@example.com"),
                password_hash=auth.hash_password("s2-pw-1"),
                role=Role.STUDENT,
            )
        )
        token, _ = auth.issue_token(user_id="u-student-2", role=Role.STUDENT)

        app = create_app()
        app.dependency_overrides[get_settings] = lambda: settings
        client = TestClient(app)
        response = client.post(
            "/api/v1/attempts",
            json={"caseId": "x", "freeText": _VALID_FREE_TEXT},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 503
        assert response.json()["code"] == "upstream_unavailable"
