"""Logging setup (structlog / stdlib), and the redaction policy log lines must obey."""
from __future__ import annotations

from cxt.infrastructure.logging.redaction import redact_host, redact_url

__all__ = ["redact_host", "redact_url"]
