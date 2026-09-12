"""MedGemmaAnalyzer — IImageAnalyzer implementation that orchestrates
the MedGemma client + the output parser.

The use case depends only on `IImageAnalyzer`; this is the adapter
that wires the prompts, the tunnel call, and the JSON parsing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from cxt.domain.ports.image_analyzer import ImageAnalysis
from cxt.domain.ports.medgemma_client import IMedGemmaClient
from cxt.domain.value_objects.cxr_image_ref import CxrImageRef
from cxt.shared.errors import UpstreamUnavailableError, ValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MedGemmaAnalyzer:
    client: IMedGemmaClient

    def analyze(self, *, image: CxrImageRef, prompt: str, system: str = "") -> ImageAnalysis:
        try:
            model_output = self.client.generate(
                image=image,
                prompt=prompt,
                system=system,
            )
        except (UpstreamUnavailableError, ValidationError):
            raise
        except Exception as exc:
            raise UpstreamUnavailableError(f"MedGemma call failed: {exc}") from exc

        if not model_output.text.strip():
            raise UpstreamUnavailableError("MedGemma returned an empty response.")

        return ImageAnalysis(raw_text=model_output.text, model_output=model_output)
