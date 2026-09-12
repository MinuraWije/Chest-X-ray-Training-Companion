"""Tests for the Phase 1 evaluation harness.

Stubs cover all five Phase 1 metrics (structure validity, finding
recognition, per-finding breakdown, latency percentiles, token usage)
plus the JSON + Markdown report shape.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

from cxt.domain.entities.case import Case
from cxt.domain.entities.finding import FindingCategory
from cxt.infrastructure import eval as eval_module
from cxt.infrastructure.data.manifest_repository import ManifestRepository
from cxt.infrastructure.eval import (
    DEFAULT_EVAL_PROMPT_VERSION,
    EVAL_PROMPT_SETS,
    EVAL_PROMPT_VERSION,
    EvalPromptSet,
    EvalReport,
    eval_prompt_fingerprint,
    eval_prompt_version,
    render_markdown_summary,
    resolve_prompt_set,
    run_evaluation,
    select_eval_cases,
    write_json_report,
    write_markdown_summary,
)
from cxt.infrastructure.ml.output_parser import OutputParser
from cxt.shared.errors import ValidationError
from tests.support import StubAnalyzer, StubCaseRepo, UnavailableAnalyzer, make_case

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cases() -> list[Case]:
    return [
        make_case("case_001", FindingCategory.PNEUMOTHORAX),
        make_case("case_002", FindingCategory.PNEUMOTHORAX),
        make_case("case_003", FindingCategory.PLEURAL_EFFUSION),
        make_case("case_004", FindingCategory.CARDIOMEGALY),
        make_case("case_005", FindingCategory.NO_FINDING),
    ]


# ---------------------------------------------------------------------------
# run_evaluation
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    def test_perfect_recognition(self) -> None:
        cases = [
            make_case("c-ptx", FindingCategory.PNEUMOTHORAX),
            make_case("c-eff", FindingCategory.PLEURAL_EFFUSION),
            make_case("c-no", FindingCategory.NO_FINDING),
        ]
        repo = StubCaseRepo(cases)
        analyzer = StubAnalyzer(default_category="no_finding", mapping=
            {
                "/images/c-ptx.png": "pneumothorax",
                "/images/c-eff.png": "pleural_effusion",
                "/images/c-no.png": "no_finding",
            }
        )
        parser = OutputParser()

        report = run_evaluation(
            case_repository=repo,
            analyzer=analyzer,
            parser=parser,
            max_cases=10,
            runtime_label="dry_run",
        )

        assert report.total_cases == 3
        assert report.parse_succeeded == 3
        assert report.parse_failed == 0
        assert report.structure_validity == 1.0
        assert report.finding_recognition == 1.0
        for stats in report.per_finding_recognition.values():
            assert stats["accuracy"] == 1.0
        assert report.latency_p50_ms > 0
        assert report.latency_p95_ms > 0
        assert report.total_prompt_tokens > 0
        assert report.total_output_tokens > 0
        assert len(analyzer.calls) == 3

    def test_mixed_recognition(self) -> None:
        cases = [
            make_case("c-ptx", FindingCategory.PNEUMOTHORAX),
            make_case("c-eff", FindingCategory.PLEURAL_EFFUSION),
            make_case("c-no", FindingCategory.NO_FINDING),
        ]
        repo = StubCaseRepo(cases)
        analyzer = StubAnalyzer(default_category="no_finding", mapping=
            {
                "/images/c-ptx.png": "pneumothorax",  # correct
                "/images/c-eff.png": "no_finding",  # wrong
                "/images/c-no.png": "no_finding",  # correct
            }
        )
        parser = OutputParser()

        report = run_evaluation(
            case_repository=repo,
            analyzer=analyzer,
            parser=parser,
            max_cases=10,
            runtime_label="dry_run",
        )

        assert report.finding_recognition == pytest.approx(2 / 3)
        assert report.per_finding_recognition["pneumothorax"]["accuracy"] == 1.0
        assert report.per_finding_recognition["pleural_effusion"]["accuracy"] == 0.0
        assert report.per_finding_recognition["no_finding"]["accuracy"] == 1.0

    def test_analyzer_failure_marked_as_error(self) -> None:
        cases = [make_case("c-1", FindingCategory.PNEUMOTHORAX)]
        repo = StubCaseRepo(cases)
        report = run_evaluation(
            case_repository=repo,
            analyzer=UnavailableAnalyzer(),
            parser=OutputParser(),
            max_cases=10,
            runtime_label="dry_run",
        )
        assert report.total_cases == 1
        assert report.parse_succeeded == 0
        assert report.finding_recognition == 0.0
        sample = report.samples[0]
        assert sample.parse_succeeded is False
        assert sample.error is not None
        assert "analyze" in sample.error

    def test_empty_split_produces_zero_report(self) -> None:
        report = run_evaluation(
            case_repository=StubCaseRepo([]),
            analyzer=StubAnalyzer(default_category="no_finding"),
            parser=OutputParser(),
            max_cases=10,
            runtime_label="dry_run",
        )
        assert report.total_cases == 0
        assert report.structure_validity == 0.0
        assert report.finding_recognition == 0.0
        assert report.latency_mean_ms == 0.0
        assert report.per_finding_recognition == {}

    def test_caps_at_max_cases(self) -> None:
        cases = [make_case(f"c-{i:03d}", FindingCategory.PNEUMOTHORAX) for i in range(10)]
        repo = StubCaseRepo(cases)
        analyzer = StubAnalyzer(default_category="no_finding")
        report = run_evaluation(
            case_repository=repo,
            analyzer=analyzer,
            parser=OutputParser(),
            max_cases=3,
            runtime_label="dry_run",
        )
        assert report.total_cases == 3
        assert len(analyzer.calls) == 3


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------


class TestWriteJsonReport:
    def test_round_trip(self, tmp_path: Path) -> None:
        cases = [
            make_case("c-1", FindingCategory.PNEUMOTHORAX),
            make_case("c-2", FindingCategory.NO_FINDING),
        ]
        repo = StubCaseRepo(cases)
        analyzer = StubAnalyzer(default_category="no_finding", mapping=
            {
                "/images/c-1.png": "pneumothorax",
                "/images/c-2.png": "no_finding",
            }
        )
        report = run_evaluation(
            case_repository=repo,
            analyzer=analyzer,
            parser=OutputParser(),
            max_cases=10,
            runtime_label="dry_run",
        )
        path = tmp_path / "report.json"
        write_json_report(report, path)
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["runtime"] == "dry_run"
        assert loaded["total_cases"] == 2
        assert loaded["structure_validity"] == 1.0
        assert "samples" in loaded
        assert len(loaded["samples"]) == 2
        sample = loaded["samples"][0]
        assert {"case_id", "ground_truth_category", "predicted_category",
                "parse_succeeded", "latency_ms", "prompt_tokens", "output_tokens",
                "error"}.issubset(sample.keys())

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        report = EvalReport(
            runtime="dry_run",
            total_cases=0,
            parse_succeeded=0,
            parse_failed=0,
            structure_validity=0.0,
            finding_recognition=0.0,
            per_finding_recognition={},
            latency_mean_ms=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            total_prompt_tokens=0,
            total_output_tokens=0,
            samples=(),
        )
        path = tmp_path / "nested" / "dir" / "report.json"
        write_json_report(report, path)
        assert path.exists()


class TestMarkdownSummary:
    def test_contains_headline_metrics(self) -> None:
        cases = [
            make_case("c-1", FindingCategory.PNEUMOTHORAX),
            make_case("c-2", FindingCategory.NO_FINDING),
        ]
        repo = StubCaseRepo(cases)
        analyzer = StubAnalyzer(default_category="no_finding", mapping={"/images/c-1.png": "pneumothorax", "/images/c-2.png": "no_finding"})
        report = run_evaluation(
            case_repository=repo,
            analyzer=analyzer,
            parser=OutputParser(),
            max_cases=10,
            runtime_label="dry_run",
        )
        text = render_markdown_summary(report)
        assert "# Phase 1 Evaluation Report" in text
        assert "Structure validity" in text
        assert "Finding recognition" in text
        assert "Per-finding recognition" in text
        assert "pneumothorax" in text
        assert "no_finding" in text
        assert "100.0%" in text  # both correct

    def test_writes_to_disk(self, tmp_path: Path) -> None:
        cases = [make_case("c-1", FindingCategory.PNEUMOTHORAX)]
        repo = StubCaseRepo(cases)
        analyzer = StubAnalyzer(default_category="no_finding", mapping={"/images/c-1.png": "pneumothorax"})
        report = run_evaluation(
            case_repository=repo,
            analyzer=analyzer,
            parser=OutputParser(),
            max_cases=10,
            runtime_label="dry_run",
        )
        path = tmp_path / "report.md"
        write_markdown_summary(report, path)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Phase 1" in content


# ---------------------------------------------------------------------------
# Latency percentiles
# ---------------------------------------------------------------------------


class TestPercentiles:
    def test_single_value(self) -> None:
        from cxt.infrastructure.eval import _percentile

        assert _percentile([42.0], 50) == 42.0
        assert _percentile([42.0], 95) == 42.0

    def test_p50_known_input(self) -> None:
        from cxt.infrastructure.eval import _percentile

        values = [float(i) for i in range(101)]  # 0..100
        assert _percentile(values, 50) == 50.0

    def test_p95_known_input(self) -> None:
        from cxt.infrastructure.eval import _percentile

        values = [float(i) for i in range(101)]  # 0..100
        assert _percentile(values, 95) == 95.0

    def test_empty_input(self) -> None:
        from cxt.infrastructure.eval import _percentile

        assert _percentile([], 50) == 0.0
        assert _percentile([], 95) == 0.0


# ---------------------------------------------------------------------------
# End-to-end with the real ManifestRepository
# ---------------------------------------------------------------------------


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Iterator[Path]:
    import csv

    rows = [
        {
            "case_id": "case_ptx",
            "image_index": "00000001_000",
            "image_path": "images/00000001_000.png",
            "finding": "pneumothorax",
            "label_source": "stub",
            "split": "test",
            "nih_labels": "Pneumothorax",
        },
        {
            "case_id": "case_no",
            "image_index": "00000002_000",
            "image_path": "images/00000002_000.png",
            "finding": "no_finding",
            "label_source": "stub",
            "split": "test",
            "nih_labels": "No Finding",
        },
    ]
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    yield path


class TestIntegrationWithManifest:
    def test_uses_manifest_split(self, manifest_path: Path) -> None:
        repo = ManifestRepository(manifest_path)
        analyzer = StubAnalyzer(default_category="no_finding", mapping=
            {
                "/images/00000001_000.png": "pneumothorax",
                "/images/00000002_000.png": "no_finding",
            }
        )
        report = run_evaluation(
            case_repository=repo,
            analyzer=analyzer,
            parser=OutputParser(),
            max_cases=10,
            runtime_label="dry_run",
        )
        assert report.total_cases == 2
        assert report.finding_recognition == 1.0


# ---------------------------------------------------------------------------
# Provenance: which model and which prompt produced the numbers (NFR-4)
# ---------------------------------------------------------------------------


class TestReportProvenance:
    def test_records_the_model_the_analyzer_reported(self) -> None:
        repo = StubCaseRepo([make_case("c1", FindingCategory.PNEUMOTHORAX)])
        report = run_evaluation(
            case_repository=repo,
            analyzer=StubAnalyzer(model="google/medgemma-1.5-4b-it"),
            parser=OutputParser(),
            max_cases=10,
            runtime_label="kaggle_tunnel",
        )
        assert report.model == "google/medgemma-1.5-4b-it"

    def test_mixed_models_are_all_reported_not_silently_picked(self) -> None:
        cases = [
            make_case("c1", FindingCategory.PNEUMOTHORAX),
            make_case("c2", FindingCategory.PNEUMOTHORAX),
        ]
        report = run_evaluation(
            case_repository=StubCaseRepo(cases),
            analyzer=StubAnalyzer(models=["model-b", "model-a"]),
            parser=OutputParser(),
            max_cases=10,
            runtime_label="kaggle_tunnel",
        )
        assert report.model == "model-a,model-b"

    def test_model_is_empty_when_no_case_produced_output(self) -> None:
        report = run_evaluation(
            case_repository=StubCaseRepo([make_case("c1", FindingCategory.PNEUMOTHORAX)]),
            analyzer=UnavailableAnalyzer(),
            parser=OutputParser(),
            max_cases=10,
            runtime_label="kaggle_tunnel",
        )
        assert report.model == ""
        assert report.prompt_version == eval_prompt_version()

    def test_prompt_version_pins_the_prompt_actually_used(self) -> None:
        report = run_evaluation(
            case_repository=StubCaseRepo([make_case("c1", FindingCategory.PNEUMOTHORAX)]),
            analyzer=StubAnalyzer(),
            parser=OutputParser(),
            max_cases=10,
            runtime_label="dry_run",
        )
        assert report.prompt_version == eval_prompt_version()
        assert report.prompt_version.startswith(f"{EVAL_PROMPT_VERSION}+")

    def test_fingerprint_changes_when_the_prompt_text_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An edit that forgets to bump EVAL_PROMPT_VERSION must still be visible."""
        before = eval_prompt_fingerprint()
        monkeypatch.setattr(eval_module, "EVAL_USER_PROMPT", "a different prompt")
        assert eval_prompt_fingerprint() != before

    def test_split_is_recorded(self) -> None:
        report = run_evaluation(
            case_repository=StubCaseRepo([make_case("c1", FindingCategory.PNEUMOTHORAX)]),
            analyzer=StubAnalyzer(),
            parser=OutputParser(),
            max_cases=10,
            runtime_label="dry_run",
            split="test",
        )
        assert report.split == "test"

    def test_empty_run_still_carries_prompt_provenance(self) -> None:
        report = run_evaluation(
            case_repository=StubCaseRepo([]),
            analyzer=StubAnalyzer(),
            parser=OutputParser(),
            max_cases=10,
            runtime_label="dry_run",
            split="test",
        )
        assert report.total_cases == 0
        assert report.prompt_version == eval_prompt_version()
        assert report.split == "test"

    def test_json_report_carries_provenance(self, tmp_path: Path) -> None:
        report = run_evaluation(
            case_repository=StubCaseRepo([make_case("c1", FindingCategory.PNEUMOTHORAX)]),
            analyzer=StubAnalyzer(model="m-1"),
            parser=OutputParser(),
            max_cases=10,
            runtime_label="kaggle_tunnel",
            split="test",
        )
        out = tmp_path / "report.json"
        write_json_report(report, out)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["model"] == "m-1"
        assert payload["prompt_version"] == eval_prompt_version()
        assert payload["split"] == "test"
        assert payload["samples"][0]["model"] == "m-1"

    def test_markdown_states_model_and_prompt_version(self) -> None:
        report = run_evaluation(
            case_repository=StubCaseRepo([make_case("c1", FindingCategory.PNEUMOTHORAX)]),
            analyzer=StubAnalyzer(model="m-1"),
            parser=OutputParser(),
            max_cases=10,
            runtime_label="kaggle_tunnel",
            split="test",
        )
        body = render_markdown_summary(report)
        assert "- Model: `m-1`" in body
        assert f"- Prompt version: `{eval_prompt_version()}`" in body
        assert "- Split: `test`" in body

    def test_markdown_says_unknown_rather_than_blank_for_a_missing_model(self) -> None:
        report = EvalReport(
            runtime="dry_run",
            total_cases=0,
            parse_succeeded=0,
            parse_failed=0,
            structure_validity=0.0,
            finding_recognition=0.0,
            per_finding_recognition={},
            latency_mean_ms=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            total_prompt_tokens=0,
            total_output_tokens=0,
        )
        body = render_markdown_summary(report)
        assert "- Model: `unknown`" in body
        assert "- Split:" not in body


# ---------------------------------------------------------------------------
# Case selection
# ---------------------------------------------------------------------------


class TestStratifiedSelection:
    """Guards the defect that produced an 11/10/6/3 sample with no negatives.

    Taking the first N manifest rows makes the per-finding table unreadable:
    a category with 3 rows reports "0.0%" that is really 0/3.
    """

    @staticmethod
    def _skewed_repo() -> StubCaseRepo:
        # Manifest order: every cardiomegaly row first, then the rest.
        cases = [make_case(f"card-{i}", FindingCategory.CARDIOMEGALY) for i in range(8)]
        cases += [make_case(f"eff-{i}", FindingCategory.PLEURAL_EFFUSION) for i in range(4)]
        cases += [make_case(f"ptx-{i}", FindingCategory.PNEUMOTHORAX) for i in range(4)]
        return StubCaseRepo(cases)

    def test_manifest_order_reproduces_the_skew(self) -> None:
        chosen = select_eval_cases(
            self._skewed_repo(), split="test", max_cases=6, stratified=False
        )
        categories = {c.ground_truth_finding.category for c in chosen}
        assert categories == {FindingCategory.CARDIOMEGALY}

    def test_stratified_spreads_across_categories(self) -> None:
        chosen = select_eval_cases(
            self._skewed_repo(), split="test", max_cases=6, stratified=True
        )
        counts = Counter(c.ground_truth_finding.category.value for c in chosen)
        assert counts == {"cardiomegaly": 2, "pleural_effusion": 2, "pneumothorax": 2}

    def test_exhausted_category_drops_out_of_the_rotation(self) -> None:
        """A category with fewer rows must not cap the whole sample."""
        cases = [make_case(f"card-{i}", FindingCategory.CARDIOMEGALY) for i in range(5)]
        cases += [make_case("solo", FindingCategory.FRACTURE)]
        chosen = select_eval_cases(StubCaseRepo(cases), split="test", max_cases=4)
        counts = Counter(c.ground_truth_finding.category.value for c in chosen)
        assert len(chosen) == 4
        assert counts["fracture"] == 1
        assert counts["cardiomegaly"] == 3

    def test_selection_is_deterministic(self) -> None:
        """Two runs must be comparable, so selection cannot be randomised."""
        first = select_eval_cases(self._skewed_repo(), split="test", max_cases=7)
        second = select_eval_cases(self._skewed_repo(), split="test", max_cases=7)
        assert [c.id for c in first] == [c.id for c in second]

    def test_respects_the_cap(self) -> None:
        chosen = select_eval_cases(self._skewed_repo(), split="test", max_cases=5)
        assert len(chosen) == 5

    def test_takes_everything_when_the_cap_exceeds_the_split(self) -> None:
        chosen = select_eval_cases(self._skewed_repo(), split="test", max_cases=99)
        assert len(chosen) == 16

    def test_filters_by_split(self) -> None:
        repo = StubCaseRepo(
            [
                make_case("t1", FindingCategory.PNEUMOTHORAX, split="test"),
                make_case("v1", FindingCategory.PNEUMOTHORAX, split="val"),
            ]
        )
        chosen = select_eval_cases(repo, split="test", max_cases=10)
        assert [c.id for c in chosen] == ["t1"]

    def test_empty_split_yields_nothing(self) -> None:
        assert select_eval_cases(StubCaseRepo([]), split="test", max_cases=10) == []

    def test_run_evaluation_stratifies_by_default(self) -> None:
        report = run_evaluation(
            case_repository=self._skewed_repo(),
            analyzer=StubAnalyzer(),
            parser=OutputParser(),
            max_cases=6,
            runtime_label="dry_run",
        )
        assert report.sampling == "stratified"
        assert set(report.per_finding_recognition) == {
            "cardiomegaly",
            "pleural_effusion",
            "pneumothorax",
        }

    def test_run_evaluation_records_manifest_order_sampling(self) -> None:
        report = run_evaluation(
            case_repository=self._skewed_repo(),
            analyzer=StubAnalyzer(),
            parser=OutputParser(),
            max_cases=6,
            runtime_label="dry_run",
            stratified=False,
        )
        assert report.sampling == "manifest_order"

    def test_markdown_reports_the_sampling_strategy(self) -> None:
        report = run_evaluation(
            case_repository=self._skewed_repo(),
            analyzer=StubAnalyzer(),
            parser=OutputParser(),
            max_cases=3,
            runtime_label="dry_run",
        )
        assert "- Sampling: `stratified`" in render_markdown_summary(report)


# ---------------------------------------------------------------------------
# Prompt sets
# ---------------------------------------------------------------------------


class TestPromptSets:
    def test_default_is_e1(self) -> None:
        assert resolve_prompt_set().version == DEFAULT_EVAL_PROMPT_VERSION
        assert resolve_prompt_set(None).qualified_version() == eval_prompt_version()

    def test_unknown_version_raises_rather_than_falling_back(self) -> None:
        """A silent fallback would let a report name a prompt it did not use."""
        with pytest.raises(ValidationError, match="Unknown eval prompt version"):
            resolve_prompt_set("e9.9")

    def test_every_registered_set_is_self_consistent(self) -> None:
        for key, prompt_set in EVAL_PROMPT_SETS.items():
            assert prompt_set.version == key
            assert prompt_set.qualified_version().startswith(f"{key}+")

    def test_versions_have_distinct_fingerprints(self) -> None:
        fingerprints = {p.fingerprint() for p in EVAL_PROMPT_SETS.values()}
        assert len(fingerprints) == len(EVAL_PROMPT_SETS)

    def test_every_set_names_all_seven_categories(self) -> None:
        for prompt_set in EVAL_PROMPT_SETS.values():
            joined = prompt_set.system + prompt_set.user
            for category in FindingCategory:
                assert category.value in joined, f"{prompt_set.version} omits {category.value}"

    def test_every_user_prompt_takes_the_clinical_context(self) -> None:
        for prompt_set in EVAL_PROMPT_SETS.values():
            assert "{clinical_context}" in prompt_set.user

    def test_selected_set_reaches_the_analyzer(self) -> None:
        analyzer = StubAnalyzer()
        run_evaluation(
            case_repository=StubCaseRepo([make_case("c1", FindingCategory.PNEUMOTHORAX)]),
            analyzer=analyzer,
            parser=OutputParser(),
            max_cases=1,
            runtime_label="dry_run",
            prompts=resolve_prompt_set("e2.0"),
        )
        _image, _prompt, system = analyzer.calls[0]
        assert system == EVAL_PROMPT_SETS["e2.0"].system

    def test_report_names_the_set_actually_used(self) -> None:
        report = run_evaluation(
            case_repository=StubCaseRepo([make_case("c1", FindingCategory.PNEUMOTHORAX)]),
            analyzer=StubAnalyzer(),
            parser=OutputParser(),
            max_cases=1,
            runtime_label="dry_run",
            prompts=resolve_prompt_set("e2.0"),
        )
        assert report.prompt_version == EVAL_PROMPT_SETS["e2.0"].qualified_version()
        assert report.prompt_version != eval_prompt_version()


class TestTwoStagePromptSets:
    """A prompt set with stage-1 fields describes the film in a separate call."""

    def test_existing_sets_are_single_stage(self) -> None:
        for version in ("e1.0", "e2.0", "e3.0", "e3.1", "e4.0"):
            assert EVAL_PROMPT_SETS[version].is_two_stage is False

    def test_e5_is_two_stage(self) -> None:
        assert EVAL_PROMPT_SETS["e5.0"].is_two_stage is True

    def test_half_configured_set_is_rejected(self) -> None:
        partial = EvalPromptSet("bad", "sys", "user", describe_system="only one")
        with pytest.raises(ValidationError):
            _ = partial.is_two_stage

    def test_fingerprint_covers_the_describe_prompts(self) -> None:
        base = EVAL_PROMPT_SETS["e5.0"]
        altered = EvalPromptSet(
            base.version,
            base.system,
            base.user,
            describe_system="a different stage 1",
            describe_user=base.describe_user,
        )
        assert altered.fingerprint() != base.fingerprint()

    def test_two_calls_are_made_per_case(self) -> None:
        analyzer = StubAnalyzer()
        run_evaluation(
            case_repository=StubCaseRepo([make_case("c1", FindingCategory.PNEUMOTHORAX)]),
            analyzer=analyzer,
            parser=OutputParser(),
            max_cases=1,
            runtime_label="dry_run",
            prompts=resolve_prompt_set("e5.0"),
        )
        assert len(analyzer.calls) == 2

    def test_stage_one_runs_first_with_the_describe_system(self) -> None:
        analyzer = StubAnalyzer()
        run_evaluation(
            case_repository=StubCaseRepo([make_case("c1", FindingCategory.PNEUMOTHORAX)]),
            analyzer=analyzer,
            parser=OutputParser(),
            max_cases=1,
            runtime_label="dry_run",
            prompts=resolve_prompt_set("e5.0"),
        )
        assert analyzer.calls[0][2] == EVAL_PROMPT_SETS["e5.0"].describe_system
        assert analyzer.calls[1][2] == EVAL_PROMPT_SETS["e5.0"].system

    def test_stage_one_output_is_interpolated_into_stage_two(self) -> None:
        analyzer = StubAnalyzer(raw_text="A distinctive stage one description.")
        run_evaluation(
            case_repository=StubCaseRepo([make_case("c1", FindingCategory.PNEUMOTHORAX)]),
            analyzer=analyzer,
            parser=OutputParser(),
            max_cases=1,
            runtime_label="dry_run",
            prompts=resolve_prompt_set("e5.0"),
        )
        assert "A distinctive stage one description." in analyzer.calls[1][1]

    def test_braces_in_the_description_do_not_break_interpolation(self) -> None:
        # Model prose routinely contains JSON-ish fragments; formatting the
        # description as part of the template would raise KeyError on them.
        analyzer = StubAnalyzer(raw_text='Findings {with} {braces} and {"a": 1}.')
        report = run_evaluation(
            case_repository=StubCaseRepo([make_case("c1", FindingCategory.PNEUMOTHORAX)]),
            analyzer=analyzer,
            parser=OutputParser(),
            max_cases=1,
            runtime_label="dry_run",
            prompts=resolve_prompt_set("e5.0"),
        )
        # The stub echoes the same text on both calls, so stage 2 legitimately
        # fails to parse. What matters here is that interpolation itself
        # survived the braces: the second call happened and carried the text.
        assert len(analyzer.calls) == 2
        assert '{"a": 1}' in analyzer.calls[1][1]
        assert report.samples[0].error == "parse: UpstreamUnavailableError"

    def test_empty_stage_one_is_reported_not_silently_classified(self) -> None:
        analyzer = StubAnalyzer(raw_text="   ")
        report = run_evaluation(
            case_repository=StubCaseRepo([make_case("c1", FindingCategory.PNEUMOTHORAX)]),
            analyzer=analyzer,
            parser=OutputParser(),
            max_cases=1,
            runtime_label="dry_run",
            prompts=resolve_prompt_set("e5.0"),
        )
        assert len(analyzer.calls) == 1
        assert report.samples[0].error == "analyze: UpstreamUnavailableError"

    def test_tokens_from_both_stages_are_counted(self) -> None:
        single = run_evaluation(
            case_repository=StubCaseRepo([make_case("c1", FindingCategory.PNEUMOTHORAX)]),
            analyzer=StubAnalyzer(),
            parser=OutputParser(),
            max_cases=1,
            runtime_label="dry_run",
            prompts=resolve_prompt_set("e4.0"),
        )
        two_stage = run_evaluation(
            case_repository=StubCaseRepo([make_case("c1", FindingCategory.PNEUMOTHORAX)]),
            analyzer=StubAnalyzer(),
            parser=OutputParser(),
            max_cases=1,
            runtime_label="dry_run",
            prompts=resolve_prompt_set("e5.0"),
        )
        assert two_stage.total_output_tokens > single.total_output_tokens


class TestLenientRecognition:
    """Recognition scored against every category the NIH labels support."""

    def test_secondary_label_counts_as_lenient_but_not_strict(self) -> None:
        # The primary reduction says pleural_effusion; the film is also
        # cardiomegaly, which is what the model names.
        case = make_case(
            "c1", FindingCategory.PLEURAL_EFFUSION, nih_labels="Cardiomegaly|Effusion"
        )
        report = run_evaluation(
            case_repository=StubCaseRepo([case]),
            analyzer=StubAnalyzer(default_category="cardiomegaly"),
            parser=OutputParser(),
            max_cases=1,
            runtime_label="dry_run",
        )
        assert report.finding_recognition == 0.0
        assert report.finding_recognition_lenient == 1.0
        assert report.samples[0].lenient_correct is True

    def test_all_true_categories_are_recorded(self) -> None:
        case = make_case(
            "c1", FindingCategory.PLEURAL_EFFUSION, nih_labels="Cardiomegaly|Effusion"
        )
        report = run_evaluation(
            case_repository=StubCaseRepo([case]),
            analyzer=StubAnalyzer(default_category="cardiomegaly"),
            parser=OutputParser(),
            max_cases=1,
            runtime_label="dry_run",
        )
        assert report.samples[0].all_true_categories == ("cardiomegaly", "pleural_effusion")

    def test_a_wrong_category_is_wrong_under_both(self) -> None:
        case = make_case("c1", FindingCategory.PLEURAL_EFFUSION, nih_labels="Effusion")
        report = run_evaluation(
            case_repository=StubCaseRepo([case]),
            analyzer=StubAnalyzer(default_category="pneumothorax"),
            parser=OutputParser(),
            max_cases=1,
            runtime_label="dry_run",
        )
        assert report.finding_recognition == 0.0
        assert report.finding_recognition_lenient == 0.0

    def test_without_nih_labels_lenient_matches_strict(self) -> None:
        cases = [
            make_case("c1", FindingCategory.PNEUMOTHORAX),
            make_case("c2", FindingCategory.CARDIOMEGALY),
        ]
        report = run_evaluation(
            case_repository=StubCaseRepo(cases),
            analyzer=StubAnalyzer(default_category="pneumothorax"),
            parser=OutputParser(),
            max_cases=2,
            runtime_label="dry_run",
        )
        assert report.finding_recognition_lenient == report.finding_recognition

    def test_unparsed_case_is_not_lenient_correct(self) -> None:
        case = make_case(
            "c1", FindingCategory.PLEURAL_EFFUSION, nih_labels="Cardiomegaly|Effusion"
        )
        report = run_evaluation(
            case_repository=StubCaseRepo([case]),
            analyzer=StubAnalyzer(raw_text="not json at all"),
            parser=OutputParser(),
            max_cases=1,
            runtime_label="dry_run",
        )
        assert report.samples[0].lenient_correct is False
        assert report.finding_recognition_lenient == 0.0

    def test_markdown_reports_both_figures(self) -> None:
        case = make_case(
            "c1", FindingCategory.PLEURAL_EFFUSION, nih_labels="Cardiomegaly|Effusion"
        )
        report = run_evaluation(
            case_repository=StubCaseRepo([case]),
            analyzer=StubAnalyzer(default_category="cardiomegaly"),
            parser=OutputParser(),
            max_cases=1,
            runtime_label="dry_run",
        )
        markdown = render_markdown_summary(report)
        assert "Finding recognition (strict):** 0.0%" in markdown
        assert "Finding recognition (lenient):** 100.0%" in markdown


class TestPatientAwareSampling:
    """Category stratification alone draws the same chest repeatedly.

    The curated subset is 300 images from 86 patients. A 12-case probe drew
    5 films of one patient, whose pacemaker and chest tube then recurred
    through the run and were read as model behaviour. Spreading across
    patients is what makes a sample's diversity real rather than nominal.
    """

    @staticmethod
    def _repo() -> StubCaseRepo:
        # One patient dominates a category, exactly as `00000013` does.
        cases = [
            make_case("a1", FindingCategory.CARDIOMEGALY, image_index="00000013_001"),
            make_case("a2", FindingCategory.CARDIOMEGALY, image_index="00000013_002"),
            make_case("a3", FindingCategory.CARDIOMEGALY, image_index="00000013_003"),
            make_case("b1", FindingCategory.CARDIOMEGALY, image_index="00000042_000"),
            make_case("c1", FindingCategory.PNEUMOTHORAX, image_index="00000013_004"),
            make_case("d1", FindingCategory.PNEUMOTHORAX, image_index="00000077_000"),
        ]
        return StubCaseRepo(cases)

    def test_a_repeat_patient_yields_to_an_unseen_one(self) -> None:
        chosen = select_eval_cases(self._repo(), split="test", max_cases=3)
        patients = [c.patient_id for c in chosen]
        assert len(set(patients)) == 3, f"expected 3 distinct patients, got {patients}"

    def test_duplicates_still_fill_the_cap_once_patients_run_out(self) -> None:
        # Diversity is a preference, not a quota: a 6-case request on 3
        # patients must still return 6 cases.
        chosen = select_eval_cases(self._repo(), split="test", max_cases=6)
        assert len(chosen) == 6

    def test_deferred_duplicates_come_last(self) -> None:
        chosen = select_eval_cases(self._repo(), split="test", max_cases=6)
        first_three = {c.patient_id for c in chosen[:3]}
        assert len(first_three) == 3

    def test_category_stratification_still_holds(self) -> None:
        counts = Counter(
            c.ground_truth_finding.category.value
            for c in select_eval_cases(self._repo(), split="test", max_cases=4)
        )
        assert counts["pneumothorax"] == 2
        assert counts["cardiomegaly"] == 2

    def test_selection_stays_deterministic(self) -> None:
        first = select_eval_cases(self._repo(), split="test", max_cases=5)
        second = select_eval_cases(self._repo(), split="test", max_cases=5)
        assert [c.id for c in first] == [c.id for c in second]

    def test_cases_without_an_image_index_are_not_treated_as_one_patient(self) -> None:
        # A hand-built corpus has no NIH index; every case must stay eligible.
        cases = [make_case(f"x{i}", FindingCategory.CARDIOMEGALY) for i in range(4)]
        chosen = select_eval_cases(StubCaseRepo(cases), split="test", max_cases=4)
        assert len(chosen) == 4

    def test_unstratified_selection_is_unaffected(self) -> None:
        chosen = select_eval_cases(
            self._repo(), split="test", max_cases=3, stratified=False
        )
        assert [c.id for c in chosen] == ["a1", "a2", "a3"]
