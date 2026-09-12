"""IAuthService port: password hashing + JWT issue/verify.

In Phase 1 the concrete impl is `argon2 + PyJWT`; the abstraction lets us
swap to a hosted auth provider later without changing use cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from cxt.domain.entities.role import Role


@dataclass(frozen=True, slots=True)
class TokenClaims:
    sub: str
    role: Role
    exp: datetime
    iat: datetime


class IAuthService(Protocol):
    def hash_password(self, plain: str) -> str: ...

    def verify_password(self, plain: str, hashed: str) -> bool: ...

    def issue_token(self, *, user_id: str, role: Role) -> tuple[str, TokenClaims]: ...

    def verify_token(self, token: str) -> TokenClaims: ...
