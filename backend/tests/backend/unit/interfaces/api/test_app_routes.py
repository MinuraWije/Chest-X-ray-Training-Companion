from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cxt.infrastructure.config.settings import MedGemmaRuntime, Settings, get_settings
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


class TestHealthzReportsTheModelRuntime:
    def test_dry_run_is_healthy_without_probing_a_tunnel(self, client: TestClient) -> None:
        payload = client.get("/api/v1/healthz").json()

        assert payload["status"] == "ok"
        assert payload["medgemmaRuntime"] == "dry_run"
        assert "tunnelReachable" not in payload

    def test_kaggle_runtime_without_a_url_is_degraded(
        self, settings: Settings, database_path: Path
    ) -> None:
        tunnel_settings = settings.model_copy(
            update={"medgemma_runtime": MedGemmaRuntime.KAGGLE_TUNNEL}
        )
        reset_dependency_caches_for_tests()
        get_settings.cache_clear()
        app = create_app()
        app.dependency_overrides[get_settings] = lambda: tunnel_settings

        payload = TestClient(app).get("/api/v1/healthz").json()

        # A configured-but-unreachable tunnel must not read as healthy.
        assert payload["status"] == "degraded"
        assert payload["tunnelReachable"] is False
        assert "MEDGEMMA_TUNNEL_URL" in payload["hint"]

        app.dependency_overrides.clear()
        reset_dependency_caches_for_tests()


def test_api_root_inventory(client: TestClient) -> None:
    response = client.get("/api/v1/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "CXT Companion API"
    assert payload["phase"]
    paths = {(r["method"], r["path"]) for r in payload["routes"]}
    expected = {
        ("GET", "/api/v1/"),
        ("GET", "/api/v1/healthz"),
        ("GET", "/api/v1/docs"),
        ("GET", "/api/v1/openapi.json"),
        ("GET", "/api/v1/cases"),
        ("GET", "/api/v1/cases/{case_id}"),
        ("POST", "/api/v1/attempts"),
        ("GET", "/api/v1/attempts/{attempt_id}"),
        ("GET", "/api/v1/attempts"),
        ("POST", "/api/v1/auth/register"),
        ("POST", "/api/v1/auth/login"),
        ("GET", "/api/v1/auth/me"),
        ("GET", "/api/v1/reviews/queue"),
        ("GET", "/api/v1/reviews/queue/{attempt_id}"),
        ("GET", "/api/v1/reviews/attempts/{attempt_id}"),
        ("POST", "/api/v1/reviews"),
    }
    assert expected.issubset(paths)
    for method, path in paths:
        assert method in ("GET", "POST")
        assert path.startswith("/api/")


def test_healthz_returns_version(client: TestClient) -> None:
    response = client.get("/api/v1/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_cases_require_auth(client: TestClient) -> None:
    response = client.get("/api/v1/cases")
    assert response.status_code == 401


def test_auth_register_open_for_students(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "a@b.com", "password": "test12345", "role": "STUDENT"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["accessToken"]
    assert body["user"]["email"] == "a@b.com"
    assert body["user"]["role"] == "STUDENT"
