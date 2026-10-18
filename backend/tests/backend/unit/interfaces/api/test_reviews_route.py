from __future__ import annotations

import csv
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
        }
    ]
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


@pytest.fixture()
def settings(tmp_path: Path, manifest_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite+pysqlite:///{tmp_path / 'cxt.db'}",
        manifest_path=manifest_path,
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


def _seed_admin(settings: Settings, email: str) -> tuple[str, str]:
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


def _register(client: TestClient, email: str, role: str = "STUDENT") -> dict:
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "long-pw-1", "role": role},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _admin_invite_educator(
    client: TestClient, admin_token: str, admin_id: str, email: str
) -> dict:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "long-pw-1",
            "role": "EDUCATOR",
            "inviteAdminId": admin_id,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _submit_attempt(
    client: TestClient, student_token: str, case_id: str = "case_aaaa1111"
) -> str:
    response = client.post(
        "/api/v1/attempts",
        json={"caseId": case_id, "freeText": "small left apical pneumothorax"},
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


class TestReviewQueue:
    def test_student_cannot_access_queue(self, client: TestClient) -> None:
        student = _register(client, "s@example.com", "STUDENT")
        response = client.get(
            "/api/v1/reviews/queue",
            headers={"Authorization": f"Bearer {student['accessToken']}"},
        )
        assert response.status_code == 403

    def test_educator_can_list_attempts(
        self, client: TestClient, settings: Settings
    ) -> None:
        student = _register(client, "s1@example.com", "STUDENT")
        attempt_id = _submit_attempt(client, student["accessToken"])

        admin_id, admin_token = _seed_admin(settings, "a1@example.com")
        educator = _admin_invite_educator(
            client, admin_token, admin_id, "e1@example.com"
        )

        response = client.get(
            "/api/v1/reviews/queue",
            headers={"Authorization": f"Bearer {educator['accessToken']}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] >= 1
        assert any(a["id"] == attempt_id for a in body["items"])

    def test_admin_can_list_attempts(
        self, client: TestClient, settings: Settings
    ) -> None:
        student = _register(client, "s2@example.com", "STUDENT")
        _submit_attempt(client, student["accessToken"])
        _admin_id, admin_token = _seed_admin(settings, "a2@example.com")
        response = client.get(
            "/api/v1/reviews/queue",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 200
        assert response.json()["total"] >= 1


class TestReviewSubmit:
    def test_educator_submits_review(
        self, client: TestClient, settings: Settings
    ) -> None:
        student = _register(client, "s3@example.com", "STUDENT")
        attempt_id = _submit_attempt(client, student["accessToken"])

        admin_id, admin_token = _seed_admin(settings, "a3@example.com")
        educator = _admin_invite_educator(
            client, admin_token, admin_id, "e3@example.com"
        )

        response = client.post(
            "/api/v1/reviews",
            json={
                "attemptId": attempt_id,
                "criteriaScores": {"finding_recognition": 4, "clarity": 5},
                "comments": "Good attempt.",
            },
            headers={"Authorization": f"Bearer {educator['accessToken']}"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["attemptId"] == attempt_id
        assert body["criteriaScores"]["finding_recognition"] == 4
        assert body["comments"] == "Good attempt."

    def test_student_cannot_submit_review(self, client: TestClient) -> None:
        student = _register(client, "s4@example.com", "STUDENT")
        attempt_id = _submit_attempt(client, student["accessToken"])
        response = client.post(
            "/api/v1/reviews",
            json={
                "attemptId": attempt_id,
                "criteriaScores": {"clarity": 4},
                "comments": "",
            },
            headers={"Authorization": f"Bearer {student['accessToken']}"},
        )
        assert response.status_code == 403

    def test_admin_submits_review(
        self, client: TestClient, settings: Settings
    ) -> None:
        student = _register(client, "s5@example.com", "STUDENT")
        attempt_id = _submit_attempt(client, student["accessToken"])
        _admin_id, admin_token = _seed_admin(settings, "a5@example.com")
        response = client.post(
            "/api/v1/reviews",
            json={
                "attemptId": attempt_id,
                "criteriaScores": {"clarity": 5},
                "comments": "ok",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 200

    def test_review_for_unknown_attempt_404(
        self, client: TestClient, settings: Settings
    ) -> None:
        _admin_id, admin_token = _seed_admin(settings, "a6@example.com")
        response = client.post(
            "/api/v1/reviews",
            json={
                "attemptId": "nope",
                "criteriaScores": {"clarity": 4},
                "comments": "",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 404

    def test_review_score_out_of_range_422(
        self, client: TestClient, settings: Settings
    ) -> None:
        student = _register(client, "s7@example.com", "STUDENT")
        attempt_id = _submit_attempt(client, student["accessToken"])
        _admin_id, admin_token = _seed_admin(settings, "a7@example.com")
        response = client.post(
            "/api/v1/reviews",
            json={
                "attemptId": attempt_id,
                "criteriaScores": {"clarity": 9},
                "comments": "",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 422


class TestReviewQueueItem:
    def test_educator_can_get_attempt_with_feedback(
        self, client: TestClient, settings: Settings
    ) -> None:
        student = _register(client, "s8@example.com", "STUDENT")
        attempt_id = _submit_attempt(client, student["accessToken"])
        _admin_id, admin_token = _seed_admin(settings, "a8@example.com")
        response = client.get(
            f"/api/v1/reviews/queue/{attempt_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == attempt_id
        assert body["feedback"] is not None

    def test_educator_sees_the_comparison_the_student_saw(
        self, client: TestClient, settings: Settings
    ) -> None:
        student = _register(client, "s10@example.com", "STUDENT")
        attempt_id = _submit_attempt(client, student["accessToken"])
        _admin_id, admin_token = _seed_admin(settings, "a10@example.com")

        as_student = client.get(
            f"/api/v1/attempts/{attempt_id}",
            headers={"Authorization": f"Bearer {student['accessToken']}"},
        ).json()
        as_educator = client.get(
            f"/api/v1/reviews/queue/{attempt_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        ).json()

        assert as_educator["comparison"] is not None
        assert as_educator["comparison"] == as_student["comparison"]

    def test_student_cannot_get_queue_item(self, client: TestClient) -> None:
        student = _register(client, "s9@example.com", "STUDENT")
        attempt_id = _submit_attempt(client, student["accessToken"])
        response = client.get(
            f"/api/v1/reviews/queue/{attempt_id}",
            headers={"Authorization": f"Bearer {student['accessToken']}"},
        )
        assert response.status_code == 403
