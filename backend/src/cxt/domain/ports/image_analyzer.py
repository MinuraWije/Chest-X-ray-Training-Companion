"""IImageAnalyzer port: orchestrates prompt building + MedGemma call + output parsing.

This is the level use cases depend on. Splitting it from `IMedGemmaClient`
keeps prompt-engineering and JSON parsing testable without a live model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cxt.domain.ports.medgemma_client import ModelOutput
from cxt.domain.value_objects.cxr_image_ref import CxrImageRef


@dataclass(frozen=True, slots=True)
class ImageAnalysis:
    """The structured result of running an image through MedGemma + the analyzer."""

    raw_text: str
    model_output: ModelOutput


class IImageAnalyzer(Protocol):
    def analyze(self, *, image: CxrImageRef, prompt: str, system: str = "") -> ImageAnalysis: ...
