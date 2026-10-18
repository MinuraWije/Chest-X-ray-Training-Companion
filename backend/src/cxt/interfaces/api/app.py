"""FastAPI application factory.

Wires:
- CORS for the Vite dev server.
- Static file mount for /images.
- Domain exception handler (DomainError -> JSON with code+message).
- Routes: healthz, /, cases, attempts, auth, reviews.

Step 5 adds the auth and review routers and replaces the
`X-Student-Id` header with JWT bearer auth.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cxt import __version__
from cxt.domain.entities.case import Case
from cxt.domain.entities.finding import FindingCategory
from cxt.domain.entities.role import Role
from cxt.domain.entities.user import User
from cxt.domain.ports.case_repository import ICaseRepository
from cxt.infrastructure.config.settings import MedGemmaRuntime, Settings, get_settings
from cxt.infrastructure.ml.kaggle_tunnel_client import KaggleTunnelMedGemmaClient
from cxt.interfaces.api.dependencies import current_user, get_case_repository
from cxt.interfaces.api.routes.attempts import router as attempts_router
from cxt.interfaces.api.routes.auth import router as auth_router
from cxt.interfaces.api.routes.reviews import router as reviews_router
from cxt.shared.errors import (
    AuthorizationError,
    DomainError,
    NotFoundError,
    UpstreamUnavailableError,
)

logger = logging.getLogger(__name__)


def _case_to_dto(case: Case, user: User) -> dict[str, object]:
    """One case as the frontend's `Case`, with the answer withheld from students.

    A student opens a case in order to interpret it, so the confirmed finding
    must not reach them before they have written an answer. Hiding it only in
    the UI is not enough: the field would still be readable in the network
    tab, and the answer comparison shown after submission would then be
    marking a student who could have read the answer first. Educators and
    admins still receive it, because curating and reviewing cases needs it.
    """
    dto: dict[str, object] = {
        "id": case.id,
        "imageUrl": case.image_url,
        "clinicalContext": case.metadata.get("clinical_context", ""),
        "patientAge": case.metadata.get("patient_age", "") or "",
        "patientSex": case.metadata.get("patient_sex", "") or "",
        "viewPosition": case.metadata.get("view_position", "") or "",
    }
    if user.role is not Role.STUDENT:
        dto["groundTruthCategory"] = case.ground_truth_finding.category.value
        dto["groundTruthLabel"] = case.ground_truth_finding.label.value
    return dto


def create_app() -> FastAPI:
    initial_settings = get_settings()
    app = FastAPI(
        title=initial_settings.app_name,
        version=initial_settings.app_version,
        docs_url="/api/v1/docs",
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(initial_settings.cors_allow_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    if initial_settings.images_dir.exists():
        app.mount(
            "/images",
            StaticFiles(directory=str(initial_settings.images_dir)),
            name="images",
        )
        logger.info("Mounted /images -> %s", initial_settings.images_dir)

    @app.get("/api/v1/healthz", tags=["meta"])
    def healthz(settings: Settings = Depends(get_settings)) -> dict[str, object]:
        health: dict[str, object] = {
            "status": "ok",
            "version": __version__,
            "medgemmaRuntime": settings.medgemma_runtime.value,
        }
        if settings.medgemma_runtime == MedGemmaRuntime.DRY_RUN:
            return health
        # A configured URL is not a reachable one — the ngrok URL changes every
        # time the notebook restarts, so actually ask.
        health["tunnelConfigured"] = bool(settings.medgemma_tunnel_url)
        health["tunnelReachable"] = (
            KaggleTunnelMedGemmaClient(settings).ping()
            if settings.medgemma_tunnel_url
            else False
        )
        if not health["tunnelReachable"]:
            health["status"] = "degraded"
            health["hint"] = (
                "MedGemma tunnel unreachable. Is the Kaggle notebook awake? "
                "Restarting it issues a new ngrok URL for MEDGEMMA_TUNNEL_URL."
            )
        return health

    @app.get("/api/v1/", tags=["meta"])
    def api_root(settings: Settings = Depends(get_settings)) -> dict[str, object]:
        routes: list[dict[str, str]] = []
        for route in app.routes:
            path = getattr(route, "path", None)
            if not isinstance(path, str) or not path.startswith("/api/"):
                continue
            methods = getattr(route, "methods", None) or set()
            for method in sorted(methods):
                if method == "HEAD":
                    continue
                routes.append({"method": method, "path": path})
        return {
            "name": settings.app_name,
            "version": __version__,
            "docs": "/api/v1/docs",
            "openapi": "/api/v1/openapi.json",
            "phase": "Phase 1 complete: MedGemma inference live via the Kaggle tunnel",
            "runtime": settings.medgemma_runtime.value,
            "tunnel_configured": bool(settings.medgemma_tunnel_url),
            "auth_required": True,
            "routes": routes,
        }

    @app.get("/api/v1/cases", tags=["cases"])
    def list_cases(
        split: str | None = Query(default=None),
        finding: FindingCategory | None = Query(default=None),
        limit: int | None = Query(default=None, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        repo: ICaseRepository | None = Depends(get_case_repository),
        user: User = Depends(current_user),
    ) -> dict[str, object]:
        if repo is None:
            raise UpstreamUnavailableError(
                "No curated case manifest found. From the backend folder run: "
                "python scripts\\dev.py real-data"
            )
        # Filtering by finding answers the question the student is meant to
        # work out: `?finding=pneumothorax` lists exactly the pneumothorax cases.
        if finding is not None and user.role is Role.STUDENT:
            raise AuthorizationError("Filtering cases by finding is not available to students.")
        cases = repo.list(split=split, finding=finding, limit=limit, offset=offset)
        total = repo.count(split=split, finding=finding)
        return {
            "items": [_case_to_dto(c, user) for c in cases],
            "total": total,
        }

    @app.get("/api/v1/cases/{case_id}", tags=["cases"])
    def get_case(
        case_id: str,
        repo: ICaseRepository | None = Depends(get_case_repository),
        user: User = Depends(current_user),
    ) -> dict[str, object]:
        if repo is None:
            raise UpstreamUnavailableError(
                "No curated case manifest found. From the backend folder run: "
                "python scripts\\dev.py real-data"
            )
        try:
            c = repo.get(case_id)
        except NotFoundError as exc:
            raise NotFoundError(str(exc)) from exc
        return _case_to_dto(c, user)

    app.include_router(auth_router)
    app.include_router(attempts_router)
    app.include_router(reviews_router)

    @app.exception_handler(DomainError)
    async def domain_error_handler(_request, exc: DomainError):  # type: ignore[no-untyped-def]
        from fastapi.responses import JSONResponse

        logger.info("Domain error: %s (%s)", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.http_status,
            content={"code": exc.code, "message": exc.message},
        )

    return app


_REPO_ROOT = Path(__file__).resolve().parents[5]


app = create_app()
