"""Tests for the case-image loaders.

The loader is what makes remote inference possible at all: the manifest emits
browser paths (`/images/x.png`) that mean nothing inside the Kaggle container.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pytest

from cxt.infrastructure.ml.image_loader import ByReferenceImageLoader, LocalImageLoader
from cxt.shared.errors import ValidationError

_PNG_BYTES = b"\x89PNG\r\n\x1a\n-not-a-real-png-but-bytes-are-bytes"


@pytest.fixture()
def images_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "images"
    directory.mkdir()
    (directory / "00000013_023.png").write_bytes(_PNG_BYTES)
    return directory


class TestLocalImageLoader:
    def test_reads_bytes_and_preserves_uri(self, images_dir: Path) -> None:
        loader = LocalImageLoader(images_dir)
        ref = loader.load("/images/00000013_023.png")

        assert ref.uri == "/images/00000013_023.png"
        assert ref.is_inline is True
        assert base64.b64decode(ref.data_b64 or "") == _PNG_BYTES
        assert ref.mime == "image/png"

    def test_accepts_a_bare_filename(self, images_dir: Path) -> None:
        loader = LocalImageLoader(images_dir)
        assert loader.load("00000013_023.png").is_inline is True

    def test_missing_file_names_the_rebuild_command(self, images_dir: Path) -> None:
        loader = LocalImageLoader(images_dir)
        with pytest.raises(ValidationError, match="real-data"):
            loader.load("/images/absent.png")

    def test_rejects_traversal_outside_the_images_dir(self, images_dir: Path) -> None:
        secret = images_dir.parent / "secret.png"
        secret.write_bytes(b"private")
        loader = LocalImageLoader(images_dir)

        with pytest.raises(ValidationError, match="outside"):
            loader.load("/images/../secret.png")

    def test_rejects_oversized_file(self, images_dir: Path) -> None:
        loader = LocalImageLoader(images_dir, max_bytes=8)
        with pytest.raises(ValidationError, match="limit"):
            loader.load("/images/00000013_023.png")

    def test_rejects_unsupported_extension(self, images_dir: Path) -> None:
        (images_dir / "notes.txt").write_text("hello")
        loader = LocalImageLoader(images_dir)
        with pytest.raises(ValidationError, match="Unsupported image type"):
            loader.load("/images/notes.txt")

    def test_rejects_empty_url(self, images_dir: Path) -> None:
        loader = LocalImageLoader(images_dir)
        with pytest.raises(ValidationError):
            loader.load("/images/")


class TestByReferenceImageLoader:
    def test_passes_the_url_through_without_touching_disk(self) -> None:
        ref = ByReferenceImageLoader().load("/images/anything.png")

        assert ref.uri == "/images/anything.png"
        assert ref.is_inline is False
        assert ref.data_b64 is None
