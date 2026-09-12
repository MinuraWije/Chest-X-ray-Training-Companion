from __future__ import annotations

import json
from pathlib import Path

from cxt.infrastructure.ml.output_parser import OutputParser

GOLDEN_DIR = Path(__file__).resolve().parent


def test_medgemma_response_to_feedback() -> None:
    raw = (GOLDEN_DIR / "medgemma_response.json").read_text(encoding="utf-8")
    expected = json.loads((GOLDEN_DIR / "feedback_dto.json").read_text(encoding="utf-8"))
    feedback = OutputParser().parse(attempt_id="test-attempt-1", raw_text=raw)
    assert feedback.attempt_id == "test-attempt-1"
    assert feedback.likely_finding.value == expected["likelyFinding"]
    assert abs(feedback.confidence.value - expected["confidence"]) < 1e-9
    assert feedback.key_signs.value == expected["keySigns"]
    assert feedback.learning_point.value == expected["learningPoint"]
    assert feedback.comparison.value == expected["comparison"]
    assert feedback.missed_clues.value == expected["missedClues"]
    assert tuple(feedback.sources) == tuple()
    assert feedback.raw["parsed"]["primary_finding"]["category"] == "pneumothorax"
    assert feedback.key_observations is not None
    assert feedback.key_observations.value == expected["keyObservations"]
    assert [
        {"label": d.label, "confidence": d.confidence.value} for d in feedback.differentials
    ] == expected["differentials"]


def test_medgemma_response_handles_markdown_fences() -> None:
    raw = (GOLDEN_DIR / "medgemma_response.json").read_text(encoding="utf-8")
    fenced = "```json\n" + raw + "\n```"
    feedback = OutputParser().parse(attempt_id="test-attempt-1", raw_text=fenced)
    assert feedback.likely_finding.value == "Right apical pneumothorax"


def test_truncated_response_still_yields_the_primary_finding() -> None:
    """Generation stopped at the token limit; the diagnosis must still survive."""
    raw = (GOLDEN_DIR / "medgemma_response_truncated.json").read_text(encoding="utf-8")
    expected = json.loads((GOLDEN_DIR / "feedback_dto.json").read_text(encoding="utf-8"))

    feedback = OutputParser().parse(attempt_id="test-attempt-1", raw_text=raw)

    assert feedback.likely_finding.value == expected["likelyFinding"]
    assert abs(feedback.confidence.value - expected["confidence"]) < 1e-9
    assert feedback.key_signs.value == expected["keySigns"]
    # The tail was lost, so the teaching point falls back rather than inventing one.
    assert feedback.learning_point.value == "(no teaching point returned)"
