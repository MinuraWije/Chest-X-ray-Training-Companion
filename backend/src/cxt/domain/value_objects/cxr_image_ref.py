"""CxrImageRef value object: how a CXR image is handed to the model runtime.

A runtime that shares a filesystem with the backend can resolve `uri`
directly. A remote runtime (the Kaggle notebook) cannot — `/images/x.png`
means nothing inside that container — so the bytes travel inline in
`data_b64`.

`uri` is always present even when bytes are attached: it is the stable
identifier used for logging, eval grouping and audit. Per NFR-6 only the
`uri` is ever logged; `data_b64` must not be.
"""
from __future__ import annotations

from dataclasses import dataclass

from cxt.shared.errors import ValidationError

_ALLOWED_MIME_TYPES: tuple[str, ...] = ("image/png", "image/jpeg")


@dataclass(frozen=True, slots=True)
class CxrImageRef:
    uri: str
    data_b64: str | None = None
    mime: str = "image/png"

    def __post_init__(self) -> None:
        if not self.uri or not isinstance(self.uri, str):
            raise ValidationError("CxrImageRef.uri must be a non-empty string.")
        if self.data_b64 is not None and (
            not isinstance(self.data_b64, str) or not self.data_b64
        ):
            raise ValidationError(
                "CxrImageRef.data_b64 must be a non-empty string when provided."
            )
        if self.mime not in _ALLOWED_MIME_TYPES:
            raise ValidationError(
                f"CxrImageRef.mime must be one of {_ALLOWED_MIME_TYPES}, got {self.mime!r}."
            )

    @property
    def is_inline(self) -> bool:
        """True when the bytes travel with the request rather than by reference."""
        return self.data_b64 is not None

    def __repr__(self) -> str:
        """Never expose the payload — repr lands in logs and tracebacks."""
        size = f"{len(self.data_b64)}b64chars" if self.data_b64 else "byref"
        return f"CxrImageRef(uri={self.uri!r}, mime={self.mime!r}, {size})"
