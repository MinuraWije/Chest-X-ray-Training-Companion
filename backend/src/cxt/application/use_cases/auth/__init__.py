"""Auth use cases: register, login, get-me.

Each is a single-method class that depends on the domain ports only. The
`RegisterUseCase` enforces:
- `STUDENT` is open (any caller can self-register).
- `EDUCATOR` and `ADMIN` require the caller to be an existing `ADMIN`.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from cxt.domain.entities.role import Role
from cxt.domain.entities.user import User
from cxt.domain.ports.auth_service import IAuthService, TokenClaims
from cxt.domain.ports.user_repository import IUserRepository
from cxt.domain.value_objects.email import Email
from cxt.shared.errors import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    ValidationError,
)

_MIN_PASSWORD_LEN = 8
_MAX_PASSWORD_LEN = 256


def _validate_email(email: str) -> Email:
    if not isinstance(email, str) or not email.strip():
        raise ValidationError("Email is required.")
    return Email(email)


def _validate_password(password: str) -> str:
    if not isinstance(password, str):
        raise ValidationError("Password is required.")
    if len(password) < _MIN_PASSWORD_LEN:
        raise ValidationError(
            f"Password must be at least {_MIN_PASSWORD_LEN} characters."
        )
    if len(password) > _MAX_PASSWORD_LEN:
        raise ValidationError(
            f"Password must be at most {_MAX_PASSWORD_LEN} characters."
        )
    return password


@dataclass(frozen=True, slots=True)
class RegisterRequest:
    email: str
    password: str
    role: Role = Role.STUDENT
    invite_admin_id: str | None = None


@dataclass(frozen=True, slots=True)
class AuthResult:
    access_token: str
    user: User
    claims: TokenClaims


class RegisterUseCase:
    def __init__(
        self,
        *,
        user_repository: IUserRepository,
        auth_service: IAuthService,
    ) -> None:
        self._users = user_repository
        self._auth = auth_service

    def execute(self, request: RegisterRequest) -> AuthResult:
        email = _validate_email(request.email)
        password = _validate_password(request.password)
        role = request.role

        if role is not Role.STUDENT:
            if request.invite_admin_id is None:
                raise AuthorizationError(
                    f"Creating a {role.value} account requires an admin invite."
                )
            inviter = self._users.get(request.invite_admin_id)
            if inviter.role is not Role.ADMIN:
                raise AuthorizationError(
                    "Only an existing ADMIN can mint an EDUCATOR or ADMIN account."
                )

        if self._users.find_by_email(email) is not None:
            raise ConflictError(f"Email already registered: {email.value}")

        user = User.new(
            id=str(uuid.uuid4()),
            email=email,
            password_hash=self._auth.hash_password(password),
            role=role,
        )
        self._users.add(user)

        token, claims = self._auth.issue_token(user_id=user.id, role=user.role)
        return AuthResult(access_token=token, user=user, claims=claims)


@dataclass(frozen=True, slots=True)
class LoginRequest:
    email: str
    password: str


class LoginUseCase:
    def __init__(
        self,
        *,
        user_repository: IUserRepository,
        auth_service: IAuthService,
    ) -> None:
        self._users = user_repository
        self._auth = auth_service

    def execute(self, request: LoginRequest) -> AuthResult:
        email = _validate_email(request.email)
        password = _validate_password(request.password)
        user = self._users.find_by_email(email)
        if user is None:
            raise AuthenticationError("Invalid email or password.")
        if not self._auth.verify_password(password, user.password_hash):
            raise AuthenticationError("Invalid email or password.")
        token, claims = self._auth.issue_token(user_id=user.id, role=user.role)
        return AuthResult(access_token=token, user=user, claims=claims)


class GetMeUseCase:
    def __init__(self, *, user_repository: IUserRepository) -> None:
        self._users = user_repository

    def execute(self, *, user_id: str) -> User:
        if not user_id:
            raise AuthenticationError("Authentication required.")
        return self._users.get(user_id)
