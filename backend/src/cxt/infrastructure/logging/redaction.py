"""Redaction helpers for anything that must never reach a log or a report.

The ngrok tunnel URL is a bearer capability, not an address: anyone holding
it can send images to the notebook until it rotates. It was being logged in
full on every startup (`get_settings`) and printed by the probe scripts,
whose output is committed under `reports/`. NFR-6 says it must not be.

`redact_url` keeps enough to debug with — the scheme and the registrable
domain, so "is the tunnel configured and is it the ngrok one?" is still
answerable — and drops the unique subdomain, which is the part that grants
access. It is deliberately total: any input, including a malformed one,
yields a safe string rather than raising inside a logging call.
"""
from __future__ import annotations

from urllib.parse import urlsplit

__all__ = ["redact_url", "redact_host"]

_UNSET = "<unset>"
_OPAQUE = "<redacted>"

# Suffixes whose final three labels are the registrable domain
# ("ngrok-free.app" is two, but "foo.com.br" style suffixes need three).
_TWO_LABEL_SUFFIXES = frozenset({"co.uk", "com.au", "com.br", "co.jp", "co.in"})


def redact_host(host: str) -> str:
    """Reduce a hostname to its registrable domain, masking the subdomain."""
    if not host:
        return _OPAQUE
    labels = host.split(".")
    if len(labels) <= 2:
        # `localhost`, `example.com` — no subdomain to hide.
        return host
    tail = ".".join(labels[-2:])
    keep = 3 if tail in _TWO_LABEL_SUFFIXES else 2
    return "***." + ".".join(labels[-keep:])


def redact_url(url: str) -> str:
    """Render a URL safe to log: scheme + registrable domain, nothing else.

    >>> redact_url("https://a1b2-34-56.ngrok-free.app/v1/analyze?k=secret")
    'https://***.ngrok-free.app'
    >>> redact_url("")
    '<unset>'
    """
    if not isinstance(url, str) or not url.strip():
        return _UNSET
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        # urlsplit raises on some malformed IPv6 literals. A logging helper
        # that can raise is worse than one that says nothing.
        return _OPAQUE
    host = parts.hostname or ""
    if not host:
        return _OPAQUE
    scheme = parts.scheme or "https"
    return f"{scheme}://{redact_host(host)}"
