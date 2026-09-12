"""Domain error hierarchy.

The application and domain layers raise these. The HTTP interface layer in
`cxt.interfaces.errors.exception_handlers` translates them into status codes
so domain code never imports FastAPI.
"""
from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain-level errors."""

    code: str = "domain_error"
    http_status: int = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(DomainError):
    code = "not_found"
    http_status = 404


class ValidationError(DomainError):
    code = "validation_error"
    http_status = 422


class AuthenticationError(DomainError):
    code = "unauthorized"
    http_status = 401


class AuthorizationError(DomainError):
    code = "forbidden"
    http_status = 403


class ConflictError(DomainError):
    code = "conflict"
    http_status = 409


class UpstreamUnavailableError(DomainError):
    """An external dependency (MedGemma, RAG) is down or returned an unusable result."""

    code = "upstream_unavailable"
    http_status = 503
