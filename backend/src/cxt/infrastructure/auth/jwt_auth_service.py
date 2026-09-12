"""argon2-cffi password hashing + PyJWT issue/verify."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from cxt.domain.entities.role import Role
from cxt.domain.ports.auth_service import IAuthService, TokenClaims
from cxt.shared.errors import AuthenticationError, ValidationError

_ROLE_FROM_STR = {
    "STUDENT": Role.STUDENT,
    "EDUCATOR": Role.EDUCATOR,
    "ADMIN": Role.ADMIN,
}


def _to_role(value: str) -> Role:
    try:
        return _ROLE_FROM_STR[value]
    except KeyError as exc:
        raise ValidationError(f"Unknown role in token: {value!r}") from exc


class JwtAuthService(IAuthService):
    def __init__(
        self,
        *,
        secret: str,
        algorithm: str,
        ttl_seconds: int,
        issuer: str,
        audience: str,
    ) -> None:
        if not secret or len(secret) < 16:
            raise ValidationError("JwtAuthService.secret must be at least 16 characters.")
        if not algorithm:
            raise ValidationError("JwtAuthService.algorithm must be a non-empty string.")
        if ttl_seconds <= 0:
            raise ValidationError("JwtAuthService.ttl_seconds must be positive.")
        self._secret = secret
        self._algorithm = algorithm
        self._ttl = timedelta(seconds=ttl_seconds)
        self._issuer = issuer
        self._audience = audience
        self._hasher = PasswordHasher()

    def hash_password(self, plain: str) -> str:
        if not isinstance(plain, str) or not plain:
            raise ValidationError("hash_password requires a non-empty string.")
        return self._hasher.hash(plain)

    def verify_password(self, plain: str, hashed: str) -> bool:
        if not isinstance(plain, str) or not isinstance(hashed, str) or not plain or not hashed:
            return False
        try:
            self._hasher.verify(hashed, plain)
        except (VerifyMismatchError, InvalidHashError):
            return False
        return True

    def issue_token(self, *, user_id: str, role: Role) -> tuple[str, TokenClaims]:
        if not user_id:
            raise ValidationError("issue_token requires a non-empty user_id.")
        iat = datetime.now(UTC)
        exp = iat + self._ttl
        payload: dict[str, object] = {
            "sub": user_id,
            "role": role.value,
            "iat": int(iat.timestamp()),
            "exp": int(exp.timestamp()),
            "iss": self._issuer,
            "aud": self._audience,
        }
        token = jwt.encode(payload, self._secret, algorithm=self._algorithm)
        return token, TokenClaims(sub=user_id, role=role, exp=exp, iat=iat)

    def verify_token(self, token: str) -> TokenClaims:
        if not isinstance(token, str) or not token:
            raise AuthenticationError("Empty or invalid bearer token.")
        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[self._algorithm],
                audience=self._audience,
                issuer=self._issuer,
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError("Token has expired.") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthenticationError(f"Invalid token: {exc}") from exc
        sub = payload.get("sub")
        role = payload.get("role")
        exp = payload.get("exp")
        iat = payload.get("iat")
        if not isinstance(sub, str) or not sub:
            raise AuthenticationError("Token missing 'sub'.")
        if not isinstance(role, str):
            raise AuthenticationError("Token missing 'role'.")
        if not isinstance(exp, int | float) or not isinstance(iat, int | float):
            raise AuthenticationError("Token missing 'iat' or 'exp'.")
        return TokenClaims(
            sub=sub,
            role=_to_role(role),
            exp=datetime.fromtimestamp(float(exp), tz=UTC),
            iat=datetime.fromtimestamp(float(iat), tz=UTC),
        )
