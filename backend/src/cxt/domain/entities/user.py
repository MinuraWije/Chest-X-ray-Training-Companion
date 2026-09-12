"""User entity: an account with a role and an argon2-hashed password."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from cxt.domain.entities.role import Role
from cxt.domain.value_objects.email import Email
from cxt.shared.errors import ValidationError


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class User:
    id: str
    email: Email
    password_hash: str
    role: Role
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValidationError("User.id must be a non-empty string.")
        if not isinstance(self.email, Email):
            raise ValidationError("User.email must be an Email value object.")
        if not self.password_hash or not isinstance(self.password_hash, str):
            raise ValidationError("User.password_hash must be a non-empty string.")
        if not isinstance(self.role, Role):
            raise ValidationError(f"User.role must be a Role, got {type(self.role).__name__}.")
        if not isinstance(self.created_at, datetime):
            raise ValidationError("User.created_at must be a datetime.")

    @classmethod
    def new(cls, *, id: str, email: Email, password_hash: str, role: Role) -> User:
        return cls(id=id, email=email, password_hash=password_hash, role=role, created_at=_utcnow())

    def can_review(self) -> bool:
        return Role.can_review(self.role)
