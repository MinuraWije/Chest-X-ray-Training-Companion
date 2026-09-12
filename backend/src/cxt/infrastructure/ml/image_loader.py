"""LocalImageLoader — turn a `/images/x.png` case URL into inline bytes.

`ManifestRepository` emits browser-relative URLs (`/images/00000013_023.png`)
because that is what the SPA needs against the `StaticFiles` mount. A remote
model runtime cannot resolve those, so this loader reads the file off disk and
base64-encodes it into a `CxrImageRef`.

The path is confined to `images_dir`: a manifest row is data, and data must not
be able to name a file outside the image directory.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Final

from cxt.domain.value_objects.cxr_image_ref import CxrImageRef
from cxt.shared.errors import ValidationError

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BYTES: Final[int] = 8 * 1024 * 1024

_MIME_BY_SUFFIX: Final[dict[str, str]] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


class LocalImageLoader:
    """Reads case images from the local processed-data directory."""

    def __init__(
        self,
        images_dir: Path,
        *,
        url_prefix: str = "/images/",
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        self._images_dir = Path(images_dir).resolve()
        self._url_prefix = url_prefix
        self._max_bytes = max_bytes

    def load(self, image_url: str) -> CxrImageRef:
        """Resolve `image_url` to a CxrImageRef carrying inline base64 bytes."""
        path = self._resolve(image_url)
        suffix = path.suffix.lower()
        mime = _MIME_BY_SUFFIX.get(suffix)
        if mime is None:
            raise ValidationError(
                f"Unsupported image type {suffix!r} for {image_url!r}; "
                f"expected one of {sorted(_MIME_BY_SUFFIX)}."
            )

        size = path.stat().st_size
        if size > self._max_bytes:
            raise ValidationError(
                f"Image {image_url!r} is {size} bytes, over the {self._max_bytes}-byte limit."
            )

        data = path.read_bytes()
        logger.debug("Loaded case image %s (%d bytes) for inline transport.", image_url, size)
        return CxrImageRef(
            uri=image_url,
            data_b64=base64.b64encode(data).decode("ascii"),
            mime=mime,
        )

    def _resolve(self, image_url: str) -> Path:
        if not image_url or not isinstance(image_url, str):
            raise ValidationError("image_url must be a non-empty string.")

        relative = image_url
        if relative.startswith(self._url_prefix):
            relative = relative[len(self._url_prefix) :]
        relative = relative.lstrip("/")
        if not relative:
            raise ValidationError(f"image_url {image_url!r} names no file.")

        candidate = (self._images_dir / relative).resolve()
        if not candidate.is_relative_to(self._images_dir):
            raise ValidationError(
                f"image_url {image_url!r} resolves outside the images directory."
            )
        if not candidate.is_file():
            raise ValidationError(
                f"Case image not found: {image_url!r}. "
                "Rebuild the manifest with: python scripts\\dev.py real-data"
            )
        return candidate


class ByReferenceImageLoader:
    """Passes the URL through untouched — for runtimes that resolve it themselves.

    Used by the dry-run analyzer, which only hashes the URI, so reading
    hundreds of PNGs off disk during tests would be pure waste.
    """

    def load(self, image_url: str) -> CxrImageRef:
        return CxrImageRef(uri=image_url)
