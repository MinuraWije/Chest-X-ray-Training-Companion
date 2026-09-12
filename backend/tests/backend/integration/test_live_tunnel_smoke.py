"""Live smoke test against a real Kaggle tunnel.

Deselected by default (`pytest -m "not slow"`), and skipped outright unless
MEDGEMMA_TUNNEL_URL is set, so it never burns Kaggle quota by accident.

Run it after pasting the notebook's ngrok URL into the environment:

    $env:MEDGEMMA_TUNNEL_URL = "https://<id>.ngrok-free.app"
    python -m pytest -m slow -q

This is the check that `DEVELOPMENT_PLAN.md:478` asks for and the evidence for
the 50%-milestone item "MedGemmaAnalyzer runs end-to-end through the Kaggle
tunnel client on at least one CXR and produces a valid Feedback".
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from cxt.domain.entities.finding import Finding, FindingCategory
from cxt.domain.value_objects.finding_label import FindingLabel
from cxt.infrastructure.config.settings import Settings
from cxt.infrastructure.ml.image_loader import LocalImageLoader
from cxt.infrastructure.ml.kaggle_tunnel_client import KaggleTunnelMedGemmaClient
from cxt.infrastructure.ml.medgemma_analyzer import MedGemmaAnalyzer
from cxt.infrastructure.ml.prompts import PromptBuilder, load_default_prompts
from cxt.infrastructure.ml.resilient_analyzer import RetryingFeedbackParser

_TUNNEL_URL = os.environ.get("MEDGEMMA_TUNNEL_URL", "")
_IMAGES_DIR = Path(__file__).resolve().parents[3].parent / "data" / "processed" / "images"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _TUNNEL_URL, reason="MEDGEMMA_TUNNEL_URL is not set"),
]


@pytest.fixture()
def settings() -> Settings:
    return Settings(medgemma_tunnel_url=_TUNNEL_URL)


def _first_image() -> Path:
    if not _IMAGES_DIR.is_dir():
        pytest.skip(f"No processed images at {_IMAGES_DIR}; run: python scripts/dev.py real-data")
    for candidate in sorted(_IMAGES_DIR.glob("*.png")):
        return candidate
    pytest.skip(f"No PNGs in {_IMAGES_DIR}")
    raise AssertionError("unreachable")


def test_tunnel_is_reachable(settings: Settings) -> None:
    assert KaggleTunnelMedGemmaClient(settings).ping() is True, (
        "Tunnel did not answer /ping. Is the notebook awake?"
    )


def test_real_cxr_produces_valid_structured_feedback(settings: Settings) -> None:
    image_path = _first_image()
    client = KaggleTunnelMedGemmaClient(settings)
    # Pay the model-load cost here rather than inside the timed call below.
    client.warm_up()

    loader = LocalImageLoader(_IMAGES_DIR, max_bytes=settings.medgemma_max_image_bytes)
    image = loader.load(f"/images/{image_path.name}")
    assert image.is_inline, "The notebook cannot resolve local paths; bytes must be inline."

    prompts = PromptBuilder(templates=load_default_prompts()).build(
        clinical_context="56-year-old male, AP view.",
        student_free_text="I think the lungs look clear.",
        # The smoke test picks an arbitrary image, so this is a stand-in for a
        # case record rather than that image's real label. It exercises the
        # round trip; it says nothing about accuracy.
        finding=Finding(
            category=FindingCategory.CARDIOMEGALY,
            label=FindingLabel("Cardiomegaly"),
        ),
    )
    result = RetryingFeedbackParser(
        MedGemmaAnalyzer(client=client),
        max_repair_attempts=settings.medgemma_max_repair_attempts,
    ).analyze_and_parse(
        attempt_id="smoke-001", image=image, prompt=prompts.user, system=prompts.system
    )

    feedback = result.feedback
    assert feedback.likely_finding.value
    assert 0.0 <= feedback.confidence.value <= 1.0
    assert feedback.key_signs.value
    assert feedback.learning_point.value
    # Real inference, not the dry-run stub.
    assert result.model_output.model
    assert result.model_output.output_tokens > 0
