"""Application settings (Pydantic Settings).

Centralised configuration loaded from environment variables / .env.
`MEDGEMMA_RUNTIME` defaults to `dry_run` so unit tests and local dev do
not burn Kaggle quota; set it to `kaggle_tunnel` for real inference. The
`Settings` instance is a process-singleton (`get_settings()`) so every
layer reads the same values — which also means a `.env` edit needs a
restart to take effect.
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from cxt.infrastructure.logging.redaction import redact_url

logger = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[5]
BACKEND_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
IMAGES_DIR: Final[Path] = PROCESSED_DIR / "images"
MANIFEST_PATH: Final[Path] = PROCESSED_DIR / "manifest.csv"
CORPUS_DIR: Final[Path] = DATA_DIR / "corpus"


class MedGemmaRuntime(str, Enum):
    KAGGLE_TUNNEL = "kaggle_tunnel"
    DRY_RUN = "dry_run"


class RetrievalRuntime(str, Enum):
    """The Phase 2 arms. One name per row of the comparison table.

    `oracle` filters on the case's confirmed finding, so it is an upper bound
    rather than a retrieval result and its precision is not reported.
    """

    NULL = "null"
    ORACLE = "oracle"
    BM25 = "bm25"
    MISCONCEPTION = "misconception"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "CXT Companion API"
    app_version: str = "0.1.0"
    environment: str = Field(default="development")

    api_v1_prefix: str = "/api/v1"
    cors_allow_origins: tuple[str, ...] = Field(
        default=("http://127.0.0.1:5173", "http://localhost:5173"),
    )

    data_dir: Path = DATA_DIR
    processed_dir: Path = PROCESSED_DIR
    images_dir: Path = IMAGES_DIR
    manifest_path: Path = MANIFEST_PATH

    medgemma_runtime: MedGemmaRuntime = MedGemmaRuntime.DRY_RUN
    medgemma_tunnel_url: str = Field(default="", description="ngrok HTTPS URL on Kaggle")
    medgemma_model: str = Field(default="google/medgemma-1.5-4b-it")
    # MedGemma 1.5 reasons before answering. An observed unquantised response
    # runs ~900 tokens: ~750 of reasoning, then the JSON. 512 truncated it
    # mid-thought, so nothing parseable was ever emitted.
    medgemma_max_new_tokens: int = Field(default=1536, ge=1, le=2048)
    # ~1500 tokens at bfloat16 on a pre-Ampere card is well over a minute.
    medgemma_timeout_seconds: float = Field(default=300.0, ge=1.0, le=600.0)
    medgemma_max_retries: int = Field(default=3, ge=1, le=10)
    # Loading a 4B checkpoint is minutes, not seconds.
    medgemma_warmup_timeout_seconds: float = Field(default=600.0, ge=1.0, le=1800.0)
    medgemma_max_repair_attempts: int = Field(default=1, ge=0, le=3)
    medgemma_max_image_bytes: int = Field(default=8 * 1024 * 1024, ge=1024)

    # Which Phase 2 retrieval arm this process runs. `null` reproduces Phase 1
    # and is the default, so an existing deployment is unchanged until asked.
    retrieval_runtime: RetrievalRuntime = RetrievalRuntime.NULL
    corpus_dir: Path = CORPUS_DIR
    retrieval_top_k: int = Field(default=3, ge=1, le=10)
    # Normalised-score floor below which the system supplies no passages at
    # all. See `retrievers.DEFAULT_SCORE_FLOOR` for why this can be absolute.
    retrieval_score_floor: float = Field(default=0.25, ge=0.0, le=1.0)

    max_eval_cases: int = Field(default=30, ge=1, le=10_000)

    auth_jwt_secret: str = Field(
        default="dev-only-not-for-production",
        description="Symmetric secret for PyJWT; replace via .env in any real deployment.",
    )
    auth_jwt_algorithm: str = Field(default="HS256")
    auth_access_token_ttl_seconds: int = Field(default=60 * 60 * 8, ge=60, le=60 * 60 * 24 * 30)
    auth_jwt_issuer: str = Field(default="cxt-companion")
    auth_jwt_audience: str = Field(default="cxt-companion")

    database_url: str = Field(default="sqlite+pysqlite:///./cxt.db")
    database_echo: bool = Field(default=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    # The tunnel URL is redacted, never printed. It is a bearer capability:
    # whoever holds it can drive the notebook until ngrok rotates it, and this
    # line runs on every process start including the ones whose output is
    # captured into `reports/`. The redacted form still answers the only
    # question this log line exists for — is a tunnel configured, and is it the
    # host I think it is.
    logger.info(
        "settings: runtime=%s tunnel=%s max_new_tokens=%d",
        settings.medgemma_runtime.value,
        redact_url(settings.medgemma_tunnel_url),
        settings.medgemma_max_new_tokens,
    )
    if os.environ.get("MEDGEMMA_DRY_RUN"):
        logger.warning(
            "MEDGEMMA_DRY_RUN is not a real setting and is being ignored. "
            "Use MEDGEMMA_RUNTIME=dry_run or MEDGEMMA_RUNTIME=kaggle_tunnel."
        )
    return settings
