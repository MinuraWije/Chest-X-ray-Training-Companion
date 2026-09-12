"""Seed the first admin (and optionally an educator) account.

The first admin is the bootstrap: there is no other way to mint an
admin or educator account through the public API (Q6 in the dev plan).
Re-running this script is safe — it is idempotent by email.

Usage:
    python scripts/seed_admin.py
    python scripts/seed_admin.py --email admin@example.com --password "long-random-string"
    python scripts/seed_admin.py --seed-educator --educator-email educator@example.com \\
        --educator-password "another-long-string"
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cxt.domain.entities.role import Role  # noqa: E402
from cxt.domain.entities.user import User  # noqa: E402
from cxt.domain.ports.auth_service import IAuthService  # noqa: E402
from cxt.domain.value_objects.email import Email  # noqa: E402
from cxt.infrastructure.auth.jwt_auth_service import JwtAuthService  # noqa: E402
from cxt.infrastructure.config.settings import Settings  # noqa: E402
from cxt.infrastructure.data.database import (  # noqa: E402
    build_engine,
    build_session_factory,
    init_database,
)
from cxt.infrastructure.data.sqlite_user_repository import (  # noqa: E402
    SqliteUserRepository,
)
from cxt.shared.errors import ConflictError  # noqa: E402

logger = logging.getLogger("seed_admin")


def _build_auth_service(settings: Settings) -> IAuthService:
    return JwtAuthService(
        secret=settings.auth_jwt_secret,
        algorithm=settings.auth_jwt_algorithm,
        ttl_seconds=settings.auth_access_token_ttl_seconds,
        issuer=settings.auth_jwt_issuer,
        audience=settings.auth_jwt_audience,
    )


def _ensure_user(
    users: SqliteUserRepository,
    auth: IAuthService,
    *,
    email: str,
    password: str,
    role: Role,
) -> tuple[User, bool]:
    existing = users.find_by_email(Email(email))
    if existing is not None:
        if existing.role is not role:
            raise SystemExit(
                f"Refusing to change role of {email}: existing={existing.role.value} requested={role.value}"
            )
        return existing, False
    user = User.new(
        id=str(uuid.uuid4()),
        email=Email(email),
        password_hash=auth.hash_password(password),
        role=role,
    )
    try:
        users.add(user)
    except ConflictError as exc:
        raise SystemExit(f"Could not seed {email}: {exc}") from exc
    return user, True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the first admin user.")
    parser.add_argument("--email", default=os.environ.get("ADMIN_EMAIL", "admin@example.com"))
    parser.add_argument(
        "--password",
        default=os.environ.get("ADMIN_PASSWORD", "change-me-in-production-12345"),
    )
    parser.add_argument("--seed-educator", action="store_true")
    parser.add_argument("--educator-email", default=os.environ.get("EDUCATOR_EMAIL", "educator@example.com"))
    parser.add_argument(
        "--educator-password",
        default=os.environ.get("EDUCATOR_PASSWORD", "change-me-in-production-12345"),
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    settings = Settings()
    engine = build_engine(settings)
    init_database(engine)
    factory = build_session_factory(engine)
    users = SqliteUserRepository(factory)
    auth = _build_auth_service(settings)

    admin, created = _ensure_user(
        users,
        auth,
        email=args.email,
        password=args.password,
        role=Role.ADMIN,
    )
    print(
        f"{'Created' if created else 'Already exists'} admin "
        f"{admin.email.value} (id={admin.id}, role={admin.role.value})"
    )

    if args.seed_educator:
        educator, created_e = _ensure_user(
            users,
            auth,
            email=args.educator_email,
            password=args.educator_password,
            role=Role.EDUCATOR,
        )
        print(
            f"{'Created' if created_e else 'Already exists'} educator "
            f"{educator.email.value} (id={educator.id}, role={educator.role.value})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
