"""SQLAlchemy engine + session factory.

`init_database` creates the schema if it doesn't exist. Tests override the
engine URL with a per-test temp file (`sqlite+pysqlite:///<tmp>/cxt.db`).
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from cxt.infrastructure.config.settings import Settings
from cxt.infrastructure.data.orm import Base


def _sqlite_file_path(database_url: str) -> str | None:
    parsed = urlparse(database_url)
    if not parsed.scheme.startswith("sqlite"):
        return None
    path = parsed.path or ""
    if path.startswith("/"):
        path = path[1:]
    return path or None


def build_engine(settings: Settings) -> Engine:
    url = settings.database_url
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        file_path = _sqlite_file_path(url)
        if file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        url,
        echo=settings.database_echo,
        future=True,
        connect_args=connect_args,
    )


def init_database(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
