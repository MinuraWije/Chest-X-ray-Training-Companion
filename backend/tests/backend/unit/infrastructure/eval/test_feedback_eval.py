"""Phase 2 feedback-evaluation metrics.

These tests assert the *definitions*, not the values. A metric that computes
something plausible but different from what the report claims it computes is
the worst outcome available here: it produces a number, the number looks
reasonable, and the thesis says something untrue about it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cxt.domain.entities.feedback import RetrievedPassage
from cxt.infrastructure.eval.feedback_eval import (
    CONTROL_ARM,
    FEEDBACK_ARMS,
    FeedbackEvalReport,
    FeedbackSample,
    compute_confabulation_rate,
    content_words,
    lexical_overlap,
    load_attempt_fixtures,
    render_feedback_summary,
    resolve_feedback_arm,
)
from cxt.shared.errors import ValidationError

FIXTURES_PATH = Path(__file__).resolve().parents[4] / "fixtures" / "student_attempts.json"

ARCHETYPES = {
    "correct",
    "missed",
    "wrong_laterality",
    "right_wrong_reason",
    "confidently_wrong",
    "vague",
}


def _sample(**overrides: object) -> FeedbackSample:
    defaults: dict[str, object] = {
        "case_id": "case-1",
        "category": "pleural_effusion",
        "archetype": "correct",
        "ok": True,
    }
    defaults.update(overrides)
    return FeedbackSample(**defaults)  # type: ignore[arg-type]


def _report(samples: tuple[FeedbackSample, ...], mode: str = "bm25") -> FeedbackEvalReport:
    return FeedbackEvalReport(
        arm=f"p3.0-{mode}",
        prompt_version="p3.0+abc",
        retrieval_mode=mode,
        corpus_version="c1.23.abc",
        model="stub",
        samples=samples,
    )


class TestArms:
    def test_five_arms_are_defined(self) -> None:
        assert set(FEEDBACK_ARMS) == {
            "p2.4",
            "p3.0-empty",
            "p3.0-oracle",
            "p3.0-bm25",
            "p3.0-misconception",
        }

    def test_the_control_arm_exists(self) -> None:
        """Confabulation is defined as a difference against it."""
        assert CONTROL_ARM in FEEDBACK_ARMS
        assert FEEDBACK_ARMS[CONTROL_ARM].retrieval_mode == "null"
        assert FEEDBACK_ARMS[CONTROL_ARM].prompt_version == "p3.0"

    def test_the_baseline_arm_is_the_frozen_prompt(self) -> None:
        assert FEEDBACK_ARMS["p2.4"].prompt_version == "p2.4"
        assert not FEEDBACK_ARMS["p2.4"].uses_retrieval

    def test_only_unfiltered_arms_report_precision(self) -> None:
        assert not FEEDBACK_ARMS["p3.0-oracle"].reports_precision
        assert FEEDBACK_ARMS["p3.0-bm25"].reports_precision

    def test_unknown_arm_raises(self) -> None:
        with pytest.raises(ValidationError, match="Unknown arm"):
            resolve_feedback_arm("p9.9-nonsense")


class TestCitationMetrics:
    def test_validity_counts_supplied_citations_only(self) -> None:
        report = _report(
            (
                _sample(cited_indices=(1, 2), hallucinated_indices=()),
                _sample(cited_indices=(1,), hallucinated_indices=(4,)),
            )
        )
        # 3 valid of 4 named.
        assert report.citation_validity == pytest.approx(0.75)

    def test_citing_nothing_scores_perfect_validity(self) -> None:
        """The exact reason citation rate must be reported beside it."""
        report = _report((_sample(), _sample()))
        assert report.citation_validity == 1.0
        assert report.citation_rate == 0.0

    def test_rate_excludes_abstentions(self) -> None:
        """A case where nothing was supplied cannot be blamed for not citing."""
        report = _report(
            (
                _sample(abstained=True),
                _sample(cited_indices=(1,)),
            )
        )
        assert report.citation_rate == 1.0

    def test_fabricated_citation_rate_is_per_sample(self) -> None:
        report = _report(
            (
                _sample(hallucinated_indices=(4,)),
                _sample(cited_indices=(1,)),
            )
        )
        assert report.hallucinated_citation_rate == pytest.approx(0.5)


class TestRetrievalMetrics:
    def test_precision_is_none_for_the_oracle(self) -> None:
        """It filters on the tag it would be scored against."""
        report = _report((_sample(relevant_at_k=3, scored_at_k=3),), mode="oracle")
        assert report.precision_at_k is None

    def test_precision_is_computed_for_bm25(self) -> None:
        report = _report(
            (
                _sample(relevant_at_k=2, scored_at_k=3),
                _sample(relevant_at_k=3, scored_at_k=3),
            )
        )
        assert report.precision_at_k == pytest.approx(5 / 6)

    def test_coverage_counts_distinct_chunks(self) -> None:
        """Exposes the degeneracy where three chunks serve every case."""
        report = _report(
            (
                _sample(retrieved_ids=("a", "b")),
                _sample(retrieved_ids=("b", "c")),
            )
        )
        assert report.corpus_coverage == 3

    def test_abstention_rate(self) -> None:
        report = _report((_sample(abstained=True), _sample(), _sample(), _sample()))
        assert report.abstention_rate == pytest.approx(0.25)


class TestLexicalOverlap:
    def test_stopwords_are_excluded(self) -> None:
        assert "the" not in content_words("The lung is the thing")
        assert "lung" in content_words("The lung is the thing")

    def test_identical_text_overlaps_fully(self) -> None:
        passage = RetrievedPassage(text="costophrenic angle blunting", source="S", id="x")
        assert lexical_overlap("costophrenic angle blunting", (passage,)) == 1.0

    def test_unrelated_text_does_not_overlap(self) -> None:
        passage = RetrievedPassage(text="costophrenic angle blunting", source="S", id="x")
        assert lexical_overlap("quantum chromodynamics lattice", (passage,)) == 0.0

    def test_no_passages_is_zero_not_an_error(self) -> None:
        assert lexical_overlap("anything at all", ()) == 0.0

    def test_report_exposes_the_null_alongside_the_measure(self) -> None:
        """Overlap without a null is uninterpretable — any two texts about
        chest radiographs share words."""
        report = _report((_sample(lexical_overlap=0.30, shuffled_overlap=0.10),))
        assert report.mean_lexical_overlap == pytest.approx(0.30)
        assert report.mean_shuffled_overlap == pytest.approx(0.10)
        assert report.overlap_lift == pytest.approx(0.20)


class TestConfabulation:
    def test_flags_vocabulary_the_control_did_not_produce(self) -> None:
        treatment = _report(
            (
                _sample(
                    feedback_text="A meniscus rises laterally along the chest wall.",
                    lexical_overlap=0.4,
                ),
            )
        )
        control = _report(
            (_sample(feedback_text="The right base is opacified."),), mode="null"
        )
        assert compute_confabulation_rate(treatment, control) == 1.0

    def test_identical_output_is_not_confabulation(self) -> None:
        text = "The right base is opacified."
        treatment = _report((_sample(feedback_text=text),))
        control = _report((_sample(feedback_text=text),), mode="null")
        assert compute_confabulation_rate(treatment, control) == 0.0

    def test_abstained_samples_are_excluded(self) -> None:
        """Nothing was supplied, so nothing could have been imported."""
        treatment = _report((_sample(feedback_text="anything", abstained=True),))
        control = _report((_sample(feedback_text="different"),), mode="null")
        assert compute_confabulation_rate(treatment, control) == 0.0

    def test_unmatched_pairs_are_skipped_not_counted(self) -> None:
        treatment = _report((_sample(case_id="case-1", feedback_text="a meniscus"),))
        control = _report((_sample(case_id="case-2"),), mode="null")
        assert compute_confabulation_rate(treatment, control) == 0.0


class TestFixtures:
    def test_fixtures_load(self) -> None:
        assert len(load_attempt_fixtures(FIXTURES_PATH)) > 0

    def test_readme_key_is_documentation_not_data(self) -> None:
        categories = {f.category for f in load_attempt_fixtures(FIXTURES_PATH)}
        assert not any(c.startswith("_") for c in categories)

    def test_every_positive_finding_has_fixtures(self) -> None:
        categories = {f.category for f in load_attempt_fixtures(FIXTURES_PATH)}
        for finding in (
            "pneumothorax",
            "pleural_effusion",
            "cardiomegaly",
            "consolidation",
            "nodule_or_mass",
            "no_finding",
        ):
            assert finding in categories, finding

    def test_every_archetype_is_represented_per_category(self) -> None:
        """Aggregate numbers hide the interesting failures; the eval reports
        per archetype, so every category needs the full set."""
        fixtures = load_attempt_fixtures(FIXTURES_PATH)
        by_category: dict[str, set[str]] = {}
        for fixture in fixtures:
            by_category.setdefault(fixture.category, set()).add(fixture.archetype)
        for category, archetypes in by_category.items():
            assert archetypes == ARCHETYPES, f"{category}: {archetypes ^ ARCHETYPES}"

    def test_answers_meet_the_api_minimum_length(self) -> None:
        """attempts.py rejects anything under 20 characters, so a fixture below
        that could never be submitted through the real path."""
        for fixture in load_attempt_fixtures(FIXTURES_PATH):
            assert len(fixture.text.strip()) >= 20, fixture.archetype

    def test_every_fixture_states_what_a_tutor_should_do(self) -> None:
        for fixture in load_attempt_fixtures(FIXTURES_PATH):
            assert fixture.expects.strip(), f"{fixture.category}/{fixture.archetype}"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        (tmp_path / "empty.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValidationError, match="no fixtures"):
            load_attempt_fixtures(tmp_path / "empty.json")


class TestSummaryRendering:
    def test_prints_both_halves_of_the_citation_pair(self) -> None:
        summary = render_feedback_summary([_report((_sample(cited_indices=(1,)),))])
        assert "Cite rate" in summary
        assert "Cite validity" in summary

    def test_marks_oracle_precision_as_not_applicable(self) -> None:
        summary = render_feedback_summary(
            [_report((_sample(relevant_at_k=3, scored_at_k=3),), mode="oracle")]
        )
        assert "n/a" in summary

    def test_shows_the_null_beside_the_overlap(self) -> None:
        summary = render_feedback_summary(
            [_report((_sample(lexical_overlap=0.3, shuffled_overlap=0.1),))]
        )
        assert "0.300 (0.100)" in summary

    def test_names_the_control_arm(self) -> None:
        summary = render_feedback_summary([_report((_sample(),))])
        assert CONTROL_ARM in summary
