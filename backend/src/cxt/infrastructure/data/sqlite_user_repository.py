"""SQLAlchemy-backed IUserRepository.

Phase 1 SQLite target; the same adapter swaps to Postgres without
domain or use-case changes.
"""
from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from cxt.domain.entities.role import Role
from cxt.domain.entities.user import User
from cxt.domain.ports.user_repository import IUserRepository
from cxt.domain.value_objects.email import Email
from cxt.infrastructure.data.database import session_scope
from cxt.infrastructure.data.orm import UserRow
from cxt.shared.errors import ConflictError, NotFoundError, ValidationError

_ROLE_FROM_DB = {
    "STUDENT": Role.STUDENT,
    "EDUCATOR": Role.EDUCATOR,
    "ADMIN": Role.ADMIN,
}


def _row_to_user(row: UserRow) -> User:
    return User(
        id=row.id,
        email=Email(row.email),
        password_hash=row.password_hash,
        role=_ROLE_FROM_DB[row.role],
        created_at=row.created_at,
    )


class SqliteUserRepository(IUserRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._factory = session_factory

    def add(self, user: User) -> None:
        with session_scope(self._factory) as session:
            session.add(
                UserRow(
                    id=user.id,
                    email=user.email.value,
                    password_hash=user.password_hash,
                    role=user.role.value,
                    created_at=user.created_at,
                )
            )

    def get(self, user_id: str) -> User:
        with session_scope(self._factory) as session:
            row = session.get(UserRow, user_id)
            if row is None:
                raise NotFoundError(f"User not found: {user_id}")
            return _row_to_user(row)

    def find_by_email(self, email: Email) -> User | None:
        with session_scope(self._factory) as session:
            stmt = select(UserRow).where(UserRow.email == email.value)
            row = session.execute(stmt).scalar_one_or_none()
            if row is None:
                return None
            return _row_to_user(row)

    def list(self) -> list[User]:
        with session_scope(self._factory) as session:
            return [_row_to_user(row) for row in _iter_users(session)]

    def try_add(self, user: User) -> None:
        """Like `add` but raises `ConflictError` on duplicate email (422-like).

        `add` raises `IntegrityError` for any DB constraint violation, including
        non-email causes; the auth use case only needs the email-uniqueness
        signal, so this wraps it.
        """
        try:
            self.add(user)
        except IntegrityError as exc:
            raise ConflictError(f"Email already registered: {user.email.value}") from exc


def _iter_users(session: Session) -> Iterable[UserRow]:
    stmt = select(UserRow).order_by(UserRow.created_at)
    return session.execute(stmt).scalars().all()


def ensure_role(value: str) -> Role:
    try:
        return _ROLE_FROM_DB[value]
    except KeyError as exc:
        raise ValidationError(f"Unknown role: {value!r}") from exc
