"""IMedGemmaClient port: minimal contract the rest of the app sees of the model.

The concrete implementation in Phase 1 is `KaggleTunnelMedGemmaClient`
(see `cxt.infrastructure.ml`). A future `GoogleApiMedGemmaClient` can be
added without changing use cases (Liskov substitution).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cxt.domain.value_objects.cxr_image_ref import CxrImageRef


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Raw model output before parsing into structured feedback."""

    text: str
    prompt_tokens: int
    output_tokens: int
    model: str = ""


class IMedGemmaClient(Protocol):
    """Calls MedGemma on a CXR image + prompt and returns the raw text."""

    def generate(self, *, image: CxrImageRef, prompt: str, system: str = "") -> ModelOutput: ...
