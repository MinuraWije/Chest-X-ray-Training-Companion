"""The case routes must not hand a student the answer before they attempt it."""
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
from cxt.interfaces.api.dependencies import reset_dependency_caches_for_tests

_ANSWER_FIELDS = ("groundTruthCategory", "groundTruthLabel")


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "case_id", "image_index", "image_path", "finding",
                "label_source", "split", "nih_labels",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "case_id": "case_aaaa1111",
            "image_index": "00000001_000",
            "image_path": "images/00000001_000.png",
            "finding": "pneumothorax",
            "label_source": "stub",
            "split": "train",
            "nih_labels": "Pneumothorax",
        })
    return Settings(
        manifest_path=manifest,
        database_url=f"sqlite+pysqlite:///{tmp_path / 'cxt.db'}",
        medgemma_runtime=MedGemmaRuntime.DRY_RUN,
        auth_jwt_secret="x" * 32,
    )


@pytest.fixture()
def headers(settings: Settings) -> dict[Role, dict[str, str]]:
    """A bearer header per role, each for a user that exists in the database."""
    auth = JwtAuthService(
        secret=settings.auth_jwt_secret,
        algorithm=settings.auth_jwt_algorithm,
        ttl_seconds=settings.auth_access_token_ttl_seconds,
        issuer=settings.auth_jwt_issuer,
        audience=settings.auth_jwt_audience,
    )
    engine = build_engine(settings)
    init_database(engine)
    users = SqliteUserRepository(build_session_factory(engine))
    result: dict[Role, dict[str, str]] = {}
    for role in (Role.STUDENT, Role.EDUCATOR):
        user_id = f"u-{role.value.lower()}"
        users.add(
            User.new(
                id=user_id,
                email=Email(f"{role.value.lower()}@example.com"),
                password_hash=auth.hash_password("long-pw-1"),
                role=role,
            )
        )
        token, _ = auth.issue_token(user_id=user_id, role=role)
        result[role] = {"Authorization": f"Bearer {token}"}
    return result


@pytest.fixture()
def client(settings: Settings, headers: dict[Role, dict[str, str]]) -> Iterator[TestClient]:
    reset_dependency_caches_for_tests()
    get_settings.cache_clear()
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()
    reset_dependency_caches_for_tests()


class TestStudentsDoNotSeeTheAnswer:
    def test_case_list_omits_the_confirmed_finding(
        self, client: TestClient, headers: dict[Role, dict[str, str]]
    ) -> None:
        body = client.get("/api/v1/cases", headers=headers[Role.STUDENT]).json()

        assert body["items"], "the fixture case should be listed"
        for item in body["items"]:
            for field in _ANSWER_FIELDS:
                assert field not in item

    def test_case_detail_omits_the_confirmed_finding(
        self, client: TestClient, headers: dict[Role, dict[str, str]]
    ) -> None:
        body = client.get("/api/v1/cases/case_aaaa1111", headers=headers[Role.STUDENT]).json()

        assert body["id"] == "case_aaaa1111"
        for field in _ANSWER_FIELDS:
            assert field not in body

    def test_filtering_by_finding_is_refused(
        self, client: TestClient, headers: dict[Role, dict[str, str]]
    ) -> None:
        # The filter would list exactly the cases with a given answer.
        response = client.get(
            "/api/v1/cases", params={"finding": "pneumothorax"}, headers=headers[Role.STUDENT]
        )

        assert response.status_code == 403


class TestEducatorsStillSeeTheAnswer:
    def test_case_detail_includes_the_confirmed_finding(
        self, client: TestClient, headers: dict[Role, dict[str, str]]
    ) -> None:
        body = client.get("/api/v1/cases/case_aaaa1111", headers=headers[Role.EDUCATOR]).json()

        assert body["groundTruthCategory"] == "pneumothorax"
        assert body["groundTruthLabel"]

    def test_filtering_by_finding_works(
        self, client: TestClient, headers: dict[Role, dict[str, str]]
    ) -> None:
        response = client.get(
            "/api/v1/cases", params={"finding": "pneumothorax"}, headers=headers[Role.EDUCATOR]
        )

        assert response.status_code == 200
        assert response.json()["total"] == 1
