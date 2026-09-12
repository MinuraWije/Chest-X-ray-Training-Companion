"""FastAPI dependency providers.

Each provider takes `Settings` via `Depends(get_settings)`, so
`app.dependency_overrides[get_settings] = lambda: test_settings`
works in tests. `get_settings` is the same function used everywhere
else; FastAPI replaces the symbol at request-resolution time.

Auth-protected routes use `Depends(current_user)` or `Depends(require_role(...))`
which read the bearer token, verify it, and look up the caller in `IUserRepository`.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Annotated, cast

from fastapi import Depends, Header
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from cxt.domain.entities.user import User
from cxt.domain.ports.attempt_repository import IAttemptRepository
from cxt.domain.ports.auth_service import IAuthService
from cxt.domain.ports.case_repository import ICaseRepository
from cxt.domain.ports.image_analyzer import IImageAnalyzer
from cxt.domain.ports.image_loader import IImageLoader
from cxt.domain.ports.retriever import IRetriever
from cxt.domain.ports.review_repository import IReviewRepository
from cxt.domain.ports.user_repository import IUserRepository
from cxt.infrastructure.auth.jwt_auth_service import JwtAuthService
from cxt.infrastructure.config.settings import MedGemmaRuntime, Settings, get_settings
from cxt.infrastructure.data.database import (
    build_engine,
    build_session_factory,
    init_database,
)
from cxt.infrastructure.data.manifest_repository import ManifestRepository
from cxt.infrastructure.data.sqlite_attempt_repository import SqliteAttemptRepository
from cxt.infrastructure.data.sqlite_review_repository import SqliteReviewRepository
from cxt.infrastructure.data.sqlite_user_repository import SqliteUserRepository
from cxt.infrastructure.ml.dry_run_analyzer import DryRunMedGemmaAnalyzer
from cxt.infrastructure.ml.image_loader import ByReferenceImageLoader, LocalImageLoader
from cxt.infrastructure.ml.kaggle_tunnel_client import KaggleTunnelMedGemmaClient
from cxt.infrastructure.ml.medgemma_analyzer import MedGemmaAnalyzer
from cxt.infrastructure.retrieval.null_retriever import NullRetriever
from cxt.shared.errors import (
    AuthenticationError,
    AuthorizationError,
    UpstreamUnavailableError,
)

EngineCacheKey = tuple[str, bool]
SessionCacheKey = tuple[str, bool, int]


@lru_cache(maxsize=8)
def _build_engine_cached(url: str, echo: bool) -> Engine:
    return build_engine(
        Settings.model_construct(
            database_url=url,
            database_echo=echo,
        )
    )


@lru_cache(maxsize=8)
def _build_session_factory_cached(url: str, echo: bool) -> sessionmaker[Session]:
    engine = _build_engine_cached(url, echo)
    init_database(engine)
    return build_session_factory(engine)


def get_engine(settings: Settings = Depends(get_settings)) -> Engine:
    return _build_engine_cached(settings.database_url, settings.database_echo)


def get_session_factory(settings: Settings = Depends(get_settings)) -> sessionmaker[Session]:
    return _build_session_factory_cached(settings.database_url, settings.database_echo)


def get_user_repository(
    settings: Settings = Depends(get_settings),
) -> IUserRepository:
    factory = get_session_factory(settings)
    return SqliteUserRepository(factory)


def get_attempt_repository(
    settings: Settings = Depends(get_settings),
) -> IAttemptRepository:
    factory = get_session_factory(settings)
    return SqliteAttemptRepository(factory)


def get_review_repository(
    settings: Settings = Depends(get_settings),
) -> IReviewRepository:
    factory = get_session_factory(settings)
    return SqliteReviewRepository(factory)


def get_auth_service(
    settings: Settings = Depends(get_settings),
) -> IAuthService:
    return JwtAuthService(
        secret=settings.auth_jwt_secret,
        algorithm=settings.auth_jwt_algorithm,
        ttl_seconds=settings.auth_access_token_ttl_seconds,
        issuer=settings.auth_jwt_issuer,
        audience=settings.auth_jwt_audience,
    )


def get_retriever() -> IRetriever:
    return NullRetriever()


def get_case_repository(
    settings: Settings = Depends(get_settings),
) -> ICaseRepository | None:
    if not settings.manifest_path.exists():
        return None
    return ManifestRepository(settings.manifest_path)


@lru_cache(maxsize=1)
def _tunnel_client(settings: Settings) -> KaggleTunnelMedGemmaClient:
    """One client (and one httpx connection pool) per process, not per request."""
    return KaggleTunnelMedGemmaClient(settings)


def get_image_analyzer(
    settings: Settings = Depends(get_settings),
) -> IImageAnalyzer:
    if settings.medgemma_runtime == MedGemmaRuntime.DRY_RUN:
        return DryRunMedGemmaAnalyzer()
    if not settings.medgemma_tunnel_url:
        raise UpstreamUnavailableError(
            "MEDGEMMA_TUNNEL_URL is not configured but MEDGEMMA_RUNTIME=kaggle_tunnel. "
            "Set MEDGEMMA_RUNTIME=dry_run in backend/.env for offline dev, "
            "or paste the ngrok URL from your Kaggle notebook into MEDGEMMA_TUNNEL_URL."
        )
    return MedGemmaAnalyzer(client=_tunnel_client(settings))


def get_image_loader(
    settings: Settings = Depends(get_settings),
) -> IImageLoader:
    """The Kaggle notebook shares no filesystem with us, so it needs the bytes.

    The dry-run analyzer only hashes the URI, so reading PNGs off disk for it
    would be pure waste.
    """
    if settings.medgemma_runtime == MedGemmaRuntime.DRY_RUN:
        return ByReferenceImageLoader()
    return LocalImageLoader(
        settings.images_dir, max_bytes=settings.medgemma_max_image_bytes
    )


def current_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    users: IUserRepository = Depends(get_user_repository),
    auth_service: IAuthService = Depends(get_auth_service),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthenticationError("Missing or malformed Authorization header.")
    token = authorization.split(" ", 1)[1].strip()
    claims = auth_service.verify_token(token)
    return users.get(claims.sub)


def require_role(*allowed_roles: str) -> Callable[[User], User]:
    from cxt.domain.entities.role import Role

    parsed = tuple(Role(r) for r in allowed_roles)

    def _checker(user: User = Depends(current_user)) -> User:
        if user.role not in parsed:
            raise AuthorizationError(
                f"This endpoint requires one of: {[r.value for r in parsed]}."
            )
        return user

    return cast(Callable[[User], User], _checker)


def reset_dependency_caches_for_tests() -> None:
    _build_engine_cached.cache_clear()
    _build_session_factory_cached.cache_clear()
