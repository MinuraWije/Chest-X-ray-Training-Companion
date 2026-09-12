"""Auth routes: register, login, me.

`POST /api/v1/auth/register` is open for `STUDENT` self-registration.
`EDUCATOR` and `ADMIN` accounts require an existing admin's id as
`inviteAdminId` in the request body.

`POST /api/v1/auth/login` returns `{accessToken, user}` on success.
`GET  /api/v1/auth/me` requires a valid bearer token.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from cxt.application.use_cases.auth import (
    GetMeUseCase,
    LoginRequest,
    LoginUseCase,
    RegisterRequest,
    RegisterUseCase,
)
from cxt.domain.entities.role import Role
from cxt.domain.entities.user import User
from cxt.domain.ports.auth_service import IAuthService
from cxt.domain.ports.user_repository import IUserRepository
from cxt.interfaces.api.dependencies import (
    current_user,
    get_auth_service,
    get_user_repository,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class RegisterBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=256)
    role: Role = Role.STUDENT
    inviteAdminId: str | None = None


class LoginBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class UserDto(BaseModel):
    id: str
    email: str
    role: Role
    createdAt: str


class TokenResponse(BaseModel):
    accessToken: str
    user: UserDto


def _to_user_dto(user: User) -> UserDto:
    return UserDto(
        id=user.id,
        email=user.email.value,
        role=user.role,
        createdAt=user.created_at.isoformat(),
    )


def _build_register_use_case(
    users: IUserRepository = Depends(get_user_repository),
    auth: IAuthService = Depends(get_auth_service),
) -> RegisterUseCase:
    return RegisterUseCase(user_repository=users, auth_service=auth)


def _build_login_use_case(
    users: IUserRepository = Depends(get_user_repository),
    auth: IAuthService = Depends(get_auth_service),
) -> LoginUseCase:
    return LoginUseCase(user_repository=users, auth_service=auth)


def _build_get_me_use_case(
    users: IUserRepository = Depends(get_user_repository),
) -> GetMeUseCase:
    return GetMeUseCase(user_repository=users)


@router.post("/register", response_model=TokenResponse)
def register(
    body: RegisterBody,
    use_case: RegisterUseCase = Depends(_build_register_use_case),
) -> TokenResponse:
    result = use_case.execute(
        RegisterRequest(
            email=body.email,
            password=body.password,
            role=body.role,
            invite_admin_id=body.inviteAdminId,
        )
    )
    return TokenResponse(
        accessToken=result.access_token,
        user=_to_user_dto(result.user),
    )


@router.post("/login", response_model=TokenResponse)
def login(
    body: LoginBody,
    use_case: LoginUseCase = Depends(_build_login_use_case),
) -> TokenResponse:
    result = use_case.execute(LoginRequest(email=body.email, password=body.password))
    return TokenResponse(
        accessToken=result.access_token,
        user=_to_user_dto(result.user),
    )


@router.get("/me", response_model=UserDto)
def me(
    user: User = Depends(current_user),
) -> UserDto:
    return _to_user_dto(user)
