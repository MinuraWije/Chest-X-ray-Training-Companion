"""Phase 2 feedback-evaluation metrics.

These tests assert the *definitions*, not the values. A metric that computes
something plausible but different from what the report claims it computes is
the worst outcome available here: it produces a number, the number looks
reasonable, and the thesis says something untrue about it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cxt.domain.entities.case import Case
from cxt.domain.entities.feedback import RetrievedPassage
from cxt.domain.entities.finding import Finding, FindingCategory
from cxt.domain.value_objects.finding_label import FindingLabel
from cxt.infrastructure.eval.feedback_eval import (
    CONTROL_ARM,
    FEEDBACK_ARMS,
    FeedbackEvalReport,
    FeedbackSample,
    StudentAttemptFixture,
    _unsupported_findings,
    content_words,
    lexical_overlap,
    load_attempt_fixtures,
    plan_pairs,
    render_feedback_summary,
    resolve_feedback_arm,
)
from cxt.infrastructure.retrieval.corpus import Corpus, CorpusChunk
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


def _corpus_with(*, chunk_id: str, findings: tuple[str, ...], text: str) -> Corpus:
    return Corpus(
        version="test-corpus",
        chunks=(
            CorpusChunk(
                id=chunk_id,
                title=chunk_id.replace("-", " "),
                text=text,
                source="Test source",
                url="https://example.org/doc",
                licence="CC BY-NC-SA 3.0",
                licence_url="https://example.org/l",
                retrieved_on="2026-09-12",
                findings=findings,
                aspect="radiographic_sign",
            ),
        ),
    )


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

    def test_an_arm_without_a_retriever_cannot_abstain(self) -> None:
        """Abstention means retrieval ran and returned nothing above the score
        floor. `p2.4` and `p3.0-empty` have no retriever at all, so marking
        them abstained reported the baseline as declining to answer on every
        case while its feedback was full and confident — and emptied the
        denominator of cite rate and overlap, which both exclude abstentions.
        """
        assert not FEEDBACK_ARMS["p2.4"].uses_retrieval
        assert not FEEDBACK_ARMS[CONTROL_ARM].uses_retrieval
        report = _report((_sample(), _sample()), mode="null")
        assert report.abstention_rate == 0.0


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
    """Confabulation is a finding named with nothing behind it.

    The definition these tests pin replaced one that compared imported
    vocabulary against the control arm and reported ~100% on every run. Two
    faults, both fatal, and both are asserted against below: generic words
    shared with a passage were counted, and importing a passage's vocabulary is
    what grounding looks like, so the old metric flagged the behaviour the
    system is trying to produce.
    """

    def test_a_finding_no_passage_and_no_case_supports_is_flagged(self) -> None:
        report = _report((_sample(unsupported_findings=("pneumothorax",)),))
        assert report.confabulation_rate == 1.0

    def test_a_feedback_naming_nothing_unsupported_is_clean(self) -> None:
        report = _report((_sample(unsupported_findings=()),))
        assert report.confabulation_rate == 0.0

    def test_the_rate_is_per_sample_not_per_finding(self) -> None:
        """One feedback naming three unsupported findings is one confabulation,
        not three; the denominator is feedbacks."""
        report = _report(
            (
                _sample(unsupported_findings=("pneumothorax", "cardiomegaly", "nodule_or_mass")),
                _sample(unsupported_findings=()),
            )
        )
        assert report.confabulation_rate == pytest.approx(0.5)

    def test_abstained_samples_are_excluded(self) -> None:
        """Nothing was supplied, so there was nothing to be unfaithful to."""
        report = _report(
            (
                _sample(unsupported_findings=("pneumothorax",), abstained=True),
                _sample(unsupported_findings=()),
            )
        )
        assert report.confabulation_rate == 0.0

    def test_no_samples_is_zero_not_an_error(self) -> None:
        assert _report(()).confabulation_rate == 0.0


class TestUnsupportedFindings:
    """`_unsupported_findings` is the detector behind the rate above."""

    def _case(self) -> Case:
        return Case(
            id="case-1",
            image_url="x.png",
            ground_truth_finding=Finding(
                category=FindingCategory.PLEURAL_EFFUSION,
                label=FindingLabel("Pleural effusion"),
            ),
        )

    def test_the_ground_truth_finding_is_always_supported(self) -> None:
        """`_enforce_ground_truth` puts it in the feedback, so counting it as
        confabulation would flag every single sample."""
        assert (
            _unsupported_findings(
                "There is a pleural effusion at the right base.", (), self._case(), None
            )
            == ()
        )

    def test_a_finding_from_a_supplied_passage_is_supported(self) -> None:
        """A passage discussing effusion as a mimic of pneumothorax is exactly
        why the model mentions pneumothorax; that is grounding, not invention."""
        corpus = _corpus_with(
            chunk_id="mimics", findings=("pneumothorax",), text="Supine effusion mimics."
        )
        passages = (RetrievedPassage(text="Supine effusion mimics.", source="S", id="mimics"),)
        assert (
            _unsupported_findings(
                "This can resemble a pneumothorax.", passages, self._case(), corpus
            )
            == ()
        )

    def test_a_finding_nothing_supplied_is_flagged(self) -> None:
        assert _unsupported_findings(
            "There is also cardiomegaly.", (), self._case(), None
        ) == ("cardiomegaly",)

    def test_a_negated_finding_is_not_an_assertion(self) -> None:
        """"There is no cardiomegaly" names the word without claiming it.
        `named_categories` strips negated spans first; this test exists so a
        change there cannot silently turn correct hedging into confabulation."""
        assert (
            _unsupported_findings("There is no cardiomegaly.", (), self._case(), None) == ()
        )

    def test_calling_something_normal_is_not_confabulation(self) -> None:
        """Confabulation is asserting a finding nothing supports; "normal" is
        the absence of one, and mis-calling a film clean is a recognition
        error, measured elsewhere. Without this, ordinary descriptive prose
        trips the metric: "unremarkable" is on the `no_finding` alias list."""
        prose = "The cardiomediastinal silhouette is unremarkable."
        assert _unsupported_findings(prose, (), self._case(), None) == ()

    def test_the_parsers_own_scaffolding_is_not_an_assertion(self) -> None:
        """`OutputParser` synthesises "Model primary: <category> ...", so the
        category name appears in the prose whether or not the model asserted
        it. Counting the harness's words as the model's claims would flag
        samples for text the model never wrote."""
        prose = (
            "Model primary: cardiomegaly (confidence 90%). Rationale: "
            "The costophrenic angle is blunted."
        )
        assert _unsupported_findings(prose, (), self._case(), None) == ()


class TestPairPlanning:
    """What a capped run actually evaluates.

    The bug these tests exist for: the manifest is ordered by category, so the
    obvious nested loop plus `max_cases=30` evaluated five pneumothorax cases
    six ways and reported it as a five-arm evaluation of the system. Every
    Phase 2 figure produced that way described one category out of six.
    """

    def _case(self, case_id: str, category: FindingCategory) -> Case:
        return Case(
            id=case_id,
            image_url=f"{case_id}.png",
            ground_truth_finding=Finding(category=category, label=FindingLabel(category.value)),
        )

    def _fixtures(self, category: str) -> list[StudentAttemptFixture]:
        return [
            StudentAttemptFixture(category=category, archetype=a, text="x" * 30)
            for a in sorted(ARCHETYPES)
        ]

    def _setup(self, per_category: int = 5) -> tuple[list[Case], dict]:
        categories = [
            FindingCategory.PNEUMOTHORAX,
            FindingCategory.PLEURAL_EFFUSION,
            FindingCategory.CARDIOMEGALY,
        ]
        cases = [
            self._case(f"{c.value}-{i}", c)
            for c in categories
            for i in range(per_category)
        ]
        by_category = {c.value: self._fixtures(c.value) for c in categories}
        return cases, by_category

    def test_a_cap_spreads_across_categories_instead_of_truncating(self) -> None:
        cases, by_category = self._setup()
        pairs = plan_pairs(cases, by_category, 9)
        categories = [c.ground_truth_finding.category.value for c, _ in pairs]
        assert len(pairs) == 9
        assert sorted(set(categories)) == ["cardiomegaly", "pleural_effusion", "pneumothorax"]
        assert all(categories.count(c) == 3 for c in set(categories))

    def test_a_cap_spreads_across_archetypes_too(self) -> None:
        """Balanced categories are not enough: thirty `correct` answers say
        nothing about the answers that actually need feedback."""
        cases, by_category = self._setup()
        pairs = plan_pairs(cases, by_category, 9)
        assert len({f.archetype for _, f in pairs}) >= 3

    def test_no_pair_is_evaluated_twice(self) -> None:
        cases, by_category = self._setup(per_category=2)
        pairs = plan_pairs(cases, by_category, 100)
        keys = [(c.id, f.archetype) for c, f in pairs]
        assert len(keys) == len(set(keys))

    def test_it_terminates_when_every_pair_is_used(self) -> None:
        """2 cases x 6 archetypes x 3 categories = 36, and asking for more
        must stop rather than spin."""
        cases, by_category = self._setup(per_category=2)
        assert len(plan_pairs(cases, by_category, None)) == 36

    def test_a_category_without_fixtures_is_skipped(self) -> None:
        cases, by_category = self._setup()
        cases.append(self._case("fracture-0", FindingCategory.FRACTURE))
        pairs = plan_pairs(cases, by_category, None)
        assert all(c.ground_truth_finding.category != FindingCategory.FRACTURE for c, _ in pairs)

    def test_no_eligible_cases_is_empty_not_an_error(self) -> None:
        assert plan_pairs([], {}, 10) == []


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
