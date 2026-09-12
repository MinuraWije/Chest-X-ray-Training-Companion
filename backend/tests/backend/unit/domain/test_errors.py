from __future__ import annotations

from cxt.shared.errors import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    DomainError,
    NotFoundError,
    UpstreamUnavailableError,
    ValidationError,
)


def test_http_status_codes() -> None:
    assert DomainError("x").http_status == 400
    assert NotFoundError("x").http_status == 404
    assert ValidationError("x").http_status == 422
    assert AuthenticationError("x").http_status == 401
    assert AuthorizationError("x").http_status == 403
    assert ConflictError("x").http_status == 409
    assert UpstreamUnavailableError("x").http_status == 503


def test_message_round_trips() -> None:
    e = NotFoundError("user 42 not found")
    assert e.message == "user 42 not found"
    assert str(e) == "user 42 not found"
