"""Tests for the prompt-experiments helper."""
from __future__ import annotations

from cxt.domain.entities.finding import FindingCategory
from cxt.domain.ports.image_analyzer import ImageAnalysis
from cxt.domain.ports.medgemma_client import ModelOutput
from cxt.infrastructure.ml.output_parser import OutputParser
from cxt.notebooks import (
    VARIATIONS,
    evaluate_variation,
    render_results_table,
)
from tests.support import StubAnalyzer, make_case


class TestEvaluateVariation:
    def test_runs_against_all_cases(self) -> None:
        cases = [make_case("c-1", FindingCategory.PNEUMOTHORAX), make_case("c-2", FindingCategory.NO_FINDING)]
        analyzer = StubAnalyzer(default_category="no_finding", mapping=
            {"/images/c-1.png": "pneumothorax", "/images/c-2.png": "no_finding"}
        )
        summary = evaluate_variation(
            cases,
            analyzer,
            OutputParser(),
            variation_name="test",
            system_prompt="SYSTEM {clinical_context}",
            user_prompt_template="USER {clinical_context}",
        )
        assert summary.variation_name == "test"
        assert len(summary.results) == 2
        assert all(r.parse_succeeded for r in summary.results)
        assert all(r.latency_ms > 0 for r in summary.results)

    def test_captures_per_case_inputs(self) -> None:
        cases = [make_case("c-1", FindingCategory.PNEUMOTHORAX)]
        analyzer = StubAnalyzer(default_category="no_finding")
        evaluate_variation(
            cases,
            analyzer,
            OutputParser(),
            variation_name="t",
            system_prompt="sys",
            user_prompt_template="u-{clinical_context}",
        )
        image, prompt, system = analyzer.calls[0]
        assert image.uri == "/images/c-1.png"
        assert system == "sys"
        assert prompt == "u-demo context"

    def test_includes_error_info_when_parse_fails(self) -> None:
        cases = [make_case("c-1", FindingCategory.PNEUMOTHORAX)]

        class _BadAnalyzer:
            def analyze(self, **_kwargs) -> ImageAnalysis:
                return ImageAnalysis(
                    raw_text="not json",
                    model_output=ModelOutput(text="not json", prompt_tokens=1, output_tokens=1),
                )

        summary = evaluate_variation(
            cases,
            _BadAnalyzer(),
            OutputParser(),
            variation_name="t",
            system_prompt="",
            user_prompt_template="{clinical_context}",
        )
        assert summary.results[0].parse_succeeded is False
        assert "parse" in (summary.results[0].error or "")

    def test_to_dict_round_trip(self) -> None:
        cases = [make_case("c-1", FindingCategory.PNEUMOTHORAX)]
        analyzer = StubAnalyzer(default_category="no_finding", mapping={"/images/c-1.png": "pneumothorax"})
        summary = evaluate_variation(
            cases,
            analyzer,
            OutputParser(),
            variation_name="t",
            system_prompt="s",
            user_prompt_template="{clinical_context}",
        )
        d = summary.to_dict()
        assert d["variation_name"] == "t"
        assert d["system_prompt"] == "s"
        assert d["user_prompt_template"] == "{clinical_context}"
        assert len(d["results"]) == 1


class TestRenderResultsTable:
    def test_renders_markdown_table(self) -> None:
        cases = [make_case("c-1", FindingCategory.PNEUMOTHORAX)]
        analyzer = StubAnalyzer(default_category="no_finding", mapping={"/images/c-1.png": "pneumothorax"})
        summary = evaluate_variation(
            cases,
            analyzer,
            OutputParser(),
            variation_name="t",
            system_prompt="s",
            user_prompt_template="{clinical_context}",
        )
        text = render_results_table(summary)
        assert "## t" in text
        assert "case_id" in text
        assert "ground_truth" in text
        assert "pneumothorax" in text


class TestVariations:
    def test_all_variations_have_required_fields(self) -> None:
        for _name, v in VARIATIONS.items():
            assert "name" in v
            assert "system" in v
            assert "user" in v
            assert v["name"]
            assert v["user"]
