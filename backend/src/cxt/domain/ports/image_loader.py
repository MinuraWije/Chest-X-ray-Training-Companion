"""IImageLoader port: resolve a case's image URL into something the model runtime can read.

A colocated runtime resolves the URL itself; a remote one needs the bytes
inline. Which of those applies is a deployment concern, so the use case
depends on this port rather than on either strategy.
"""
from __future__ import annotations

from typing import Protocol

from cxt.domain.value_objects.cxr_image_ref import CxrImageRef


class IImageLoader(Protocol):
    def load(self, image_url: str) -> CxrImageRef: ...
