from __future__ import annotations

import pytest

from cxt.domain.value_objects.cxr_image_ref import CxrImageRef
from cxt.infrastructure.ml.dry_run_analyzer import DryRunMedGemmaAnalyzer
from cxt.infrastructure.ml.output_parser import OutputParser


@pytest.fixture()
def analyzer() -> DryRunMedGemmaAnalyzer:
    return DryRunMedGemmaAnalyzer()


@pytest.fixture()
def parser() -> OutputParser:
    return OutputParser()


def test_returns_image_analysis_with_text(analyzer: DryRunMedGemmaAnalyzer) -> None:
    analysis = analyzer.analyze(
        image=CxrImageRef(uri="https://example.com/cxr.png"),
        prompt="describe this",
        system="you are a radiologist",
    )
    assert analysis.raw_text
    assert analysis.model_output.text == analysis.raw_text
    assert analysis.model_output.prompt_tokens > 0
    assert analysis.model_output.output_tokens > 0


def test_output_parses_to_feedback(
    analyzer: DryRunMedGemmaAnalyzer, parser: OutputParser
) -> None:
    analysis = analyzer.analyze(
        image=CxrImageRef(uri="https://example.com/cxr.png"),
        prompt="describe this",
    )
    feedback = parser.parse(attempt_id="a1", raw_text=analysis.raw_text)
    assert feedback.attempt_id == "a1"
    assert feedback.likely_finding.value
    assert 0.0 <= feedback.confidence.value <= 1.0
    assert feedback.key_signs.value
    assert feedback.learning_point.value


def test_seven_categories_are_covered(
    analyzer: DryRunMedGemmaAnalyzer, parser: OutputParser
) -> None:
    """Over many seeds the dry-run analyzer should hit all six findings + no_finding."""
    categories_seen: set[str] = set()
    for i in range(60):
        analysis = analyzer.analyze(
            image=CxrImageRef(uri=f"https://example.com/cxr-{i}.png"),
            prompt="describe this",
        )
        feedback = parser.parse(attempt_id=f"a{i}", raw_text=analysis.raw_text)
        categories_seen.add(feedback.raw["parsed"]["primary_finding"]["category"])
    expected = {
        "pneumothorax",
        "pleural_effusion",
        "cardiomegaly",
        "consolidation",
        "nodule_or_mass",
        "fracture",
        "no_finding",
    }
    assert expected.issubset(categories_seen), categories_seen


def test_determinism(analyzer: DryRunMedGemmaAnalyzer) -> None:
    a = analyzer.analyze(
        image=CxrImageRef(uri="https://example.com/cxr-stable.png"),
        prompt="x",
    )
    b = analyzer.analyze(
        image=CxrImageRef(uri="https://example.com/cxr-stable.png"),
        prompt="x",
    )
    assert a.raw_text == b.raw_text
