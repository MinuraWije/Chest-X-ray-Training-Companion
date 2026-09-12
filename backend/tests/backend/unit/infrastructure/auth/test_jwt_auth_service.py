from __future__ import annotations

import pytest

from cxt.domain.entities.role import Role
from cxt.domain.entities.user import User
from cxt.domain.value_objects.email import Email
from cxt.infrastructure.auth.jwt_auth_service import JwtAuthService
from cxt.shared.errors import AuthenticationError, ValidationError


def _service(**overrides) -> JwtAuthService:  # type: ignore[no-untyped-def]
    defaults = {
        "secret": "x" * 32,
        "algorithm": "HS256",
        "ttl_seconds": 60,
        "issuer": "cxt-companion",
        "audience": "cxt-companion",
    }
    defaults.update(overrides)
    return JwtAuthService(**defaults)


class TestPasswordHashing:
    def test_hash_returns_argon2_string(self) -> None:
        h = _service().hash_password("hello-world-1")
        assert h.startswith("$argon2")

    def test_verify_accepts_correct_password(self) -> None:
        service = _service()
        h = service.hash_password("hello-world-1")
        assert service.verify_password("hello-world-1", h) is True

    def test_verify_rejects_wrong_password(self) -> None:
        service = _service()
        h = service.hash_password("hello-world-1")
        assert service.verify_password("wrong", h) is False

    def test_verify_rejects_empty(self) -> None:
        service = _service()
        h = service.hash_password("hello-world-1")
        assert service.verify_password("", h) is False
        assert service.verify_password("x", "") is False

    def test_hash_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            _service().hash_password("")


class TestJwtIssueAndVerify:
    def test_round_trip(self) -> None:
        service = _service()
        token, claims = service.issue_token(user_id="u1", role=Role.STUDENT)
        decoded = service.verify_token(token)
        assert decoded.sub == "u1"
        assert decoded.role is Role.STUDENT
        assert int(decoded.exp.timestamp()) == int(claims.exp.timestamp())
        assert int(decoded.iat.timestamp()) == int(claims.iat.timestamp())

    def test_verify_rejects_empty(self) -> None:
        with pytest.raises(AuthenticationError):
            _service().verify_token("")

    def test_verify_rejects_garbage(self) -> None:
        with pytest.raises(AuthenticationError):
            _service().verify_token("not.a.jwt")

    def test_verify_rejects_wrong_secret(self) -> None:
        token, _ = _service().issue_token(user_id="u1", role=Role.STUDENT)
        other = _service(secret="y" * 32)
        with pytest.raises(AuthenticationError):
            other.verify_token(token)

    def test_verify_rejects_expired(self) -> None:
        # ttl=0 isn't allowed, so build a token with a tiny ttl and wait.
        import time

        service = _service(ttl_seconds=1)
        token, _ = service.issue_token(user_id="u1", role=Role.STUDENT)
        time.sleep(1.2)
        with pytest.raises(AuthenticationError):
            service.verify_token(token)

    def test_verify_rejects_wrong_audience(self) -> None:
        token, _ = _service(audience="a").issue_token(user_id="u1", role=Role.STUDENT)
        with pytest.raises(AuthenticationError):
            _service(audience="b").verify_token(token)

    def test_issue_rejects_empty_user_id(self) -> None:
        with pytest.raises(ValidationError):
            _service().issue_token(user_id="", role=Role.STUDENT)


class TestConstructorValidation:
    def test_short_secret_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _service(secret="short")

    def test_zero_ttl_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _service(ttl_seconds=0)

    def test_empty_algorithm_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _service(algorithm="")


def test_user_can_review_predicate_via_constructor() -> None:
    u = User.new(
        id="u1",
        email=Email("a@b.com"),
        password_hash="h",
        role=Role.EDUCATOR,
    )
    assert u.can_review() is True
