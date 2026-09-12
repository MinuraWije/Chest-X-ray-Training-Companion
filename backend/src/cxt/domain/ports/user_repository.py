"""IUserRepository port: persistence for User entities."""
from __future__ import annotations

from typing import Protocol

from cxt.domain.entities.user import User
from cxt.domain.value_objects.email import Email


class IUserRepository(Protocol):
    def add(self, user: User) -> None: ...

    def get(self, user_id: str) -> User: ...

    def find_by_email(self, email: Email) -> User | None: ...
