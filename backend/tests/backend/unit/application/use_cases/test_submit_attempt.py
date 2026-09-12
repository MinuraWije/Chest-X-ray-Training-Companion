from __future__ import annotations

from typing import Any

import pytest

from cxt.application.use_cases.submit_attempt import (
    SubmitAttemptRequest,
    SubmitAttemptUseCase,
)
from cxt.domain.entities.case import Case
from cxt.domain.entities.feedback import Feedback
from cxt.domain.entities.finding import Finding, FindingCategory
from cxt.domain.entities.student_attempt import StudentAttempt
from cxt.domain.value_objects.finding_label import FindingLabel
from cxt.infrastructure.ml.dry_run_analyzer import DryRunMedGemmaAnalyzer
from cxt.infrastructure.ml.output_parser import OutputParser
from cxt.infrastructure.ml.prompts import PromptBuilder, load_default_prompts
from cxt.infrastructure.retrieval.null_retriever import NullRetriever
from cxt.shared.errors import NotFoundError, UpstreamUnavailableError, ValidationError
from tests.support import StubAnalyzer


def _case(id_: str = "c1") -> Case:
    return Case(
        id=id_,
        image_url="/images/c1.png",
        ground_truth_finding=Finding(
            category=FindingCategory.PNEUMOTHORAX,
            label=FindingLabel("Pneumothorax"),
        ),
        metadata={"clinical_context": "23yo M, sudden chest pain"},
    )


class _StubCaseRepo:
    def __init__(self, case: Case) -> None:
        self._case = case

    def get(self, case_id: str) -> Case:
        if case_id != self._case.id:
            raise NotFoundError(f"Case not found: {case_id}")
        return self._case

    def list(self, **_kwargs) -> list[Case]:  # type: ignore[no-untyped-def]
        return [self._case]

    def count(self, **_kwargs) -> int:  # type: ignore[no-untyped-def]
        return 1


class _StubAttemptRepo:
    def __init__(self) -> None:
        self._attempts: dict[str, StudentAttempt] = {}
        self._feedback: dict[str, Feedback] = {}

    def add(self, attempt: StudentAttempt) -> None:
        if attempt.id in self._attempts:
            raise ValidationError(f"Attempt {attempt.id} already exists.")
        self._attempts[attempt.id] = attempt

    def get(self, attempt_id: str) -> StudentAttempt:
        try:
            return self._attempts[attempt_id]
        except KeyError as exc:
            raise NotFoundError(f"Attempt not found: {attempt_id}") from exc

    def get_with_feedback(self, attempt_id: str) -> tuple[StudentAttempt, Feedback | None]:
        attempt = self.get(attempt_id)
        return attempt, self._feedback.get(attempt_id)

    def list_for_student(self, student_id: str) -> list[StudentAttempt]:
        return [a for a in self._attempts.values() if a.student_id == student_id]

    def list_for_educator_review(self) -> list[StudentAttempt]:
        return list(self._attempts.values())

    def attach_feedback(self, attempt_id: str, feedback: Feedback) -> None:
        self._feedback[attempt_id] = feedback


def _build(**overrides: Any) -> SubmitAttemptUseCase:
    repo = overrides.get("attempts", _StubAttemptRepo())
    case_repo = overrides.get("case_repo", _StubCaseRepo(_case()))
    return SubmitAttemptUseCase(
        analyzer=overrides.get("analyzer", DryRunMedGemmaAnalyzer()),
        attempt_repository=repo,
        case_repository=case_repo,
        retriever=overrides.get("retriever", NullRetriever()),
        prompt_builder=overrides.get("prompt_builder", PromptBuilder(templates=load_default_prompts())),
        output_parser=overrides.get("output_parser", OutputParser()),
    )


class TestHappyPath:
    def test_returns_attempt_and_feedback(self) -> None:
        use_case = _build()
        result = use_case.execute(
            SubmitAttemptRequest(
                case_id="c1",
                student_id="alice",
                free_text="Small left apical pneumothorax with visible visceral pleural line.",
            )
        )
        assert result.attempt.id
        assert result.attempt.case_id == "c1"
        assert result.attempt.student_id == "alice"
        assert result.attempt.free_text.startswith("Small")
        assert result.feedback.likely_finding.value
        assert 0.0 <= result.feedback.confidence.value <= 1.0
        assert result.feedback.key_signs.value
        assert result.feedback.learning_point.value
        assert result.feedback.sources == ()

    def test_persists_attempt_with_feedback(self) -> None:
        attempts = _StubAttemptRepo()
        use_case = _build(attempts=attempts)
        result = use_case.execute(
            SubmitAttemptRequest(case_id="c1", student_id="alice", free_text="x y z")
        )
        stored = attempts.get(result.attempt.id)
        assert stored.id == result.attempt.id

    def test_attaches_feedback_to_repo(self) -> None:
        attempts = _StubAttemptRepo()
        use_case = _build(attempts=attempts)
        result = use_case.execute(
            SubmitAttemptRequest(case_id="c1", student_id="alice", free_text="x y z")
        )
        _stored, feedback = attempts.get_with_feedback(result.attempt.id)
        assert feedback is not None
        assert feedback.likely_finding.value == result.feedback.likely_finding.value

    def test_generates_unique_attempt_ids(self) -> None:
        use_case = _build()
        ids = set()
        for _ in range(5):
            r = use_case.execute(
                SubmitAttemptRequest(case_id="c1", student_id="alice", free_text="x y z")
            )
            ids.add(r.attempt.id)
        assert len(ids) == 5


class TestValidation:
    def test_blank_student_id_rejected(self) -> None:
        use_case = _build()
        with pytest.raises(ValidationError):
            use_case.execute(
                SubmitAttemptRequest(case_id="c1", student_id="", free_text="x y z")
            )

    def test_blank_free_text_rejected(self) -> None:
        use_case = _build()
        with pytest.raises(ValidationError):
            use_case.execute(
                SubmitAttemptRequest(case_id="c1", student_id="alice", free_text="   ")
            )


class TestCaseRepoErrors:
    def test_unknown_case_raises_not_found(self) -> None:
        use_case = _build()
        with pytest.raises(NotFoundError):
            use_case.execute(
                SubmitAttemptRequest(case_id="nope", student_id="alice", free_text="x y z")
            )


class TestAnalyzerErrors:
    def test_upstream_error_propagates(self) -> None:
        class _BoomAnalyzer:
            def analyze(self, **_kwargs: Any) -> None:
                raise UpstreamUnavailableError("model offline")

        use_case = _build(analyzer=_BoomAnalyzer())
        with pytest.raises(UpstreamUnavailableError):
            use_case.execute(
                SubmitAttemptRequest(case_id="c1", student_id="alice", free_text="x y z")
            )


class TestGroundTruthIsSupplied:
    """The case's verified finding is handed to the model, not asked for.

    The model's own recognition accuracy sits near chance (16-25% across six
    eval prompt sets), so feedback built on its guess would misjudge most
    students. These tests pin the contract that keeps that guess out of the
    student-facing path.
    """

    def _run(self, analyzer: StubAnalyzer) -> Feedback:
        use_case = _build(analyzer=analyzer)
        return use_case.execute(
            SubmitAttemptRequest(
                case_id="c1",
                student_id="alice",
                free_text="I think there is a small left apical pneumothorax.",
            )
        ).feedback

    def test_prompts_carry_the_confirmed_category(self) -> None:
        analyzer = StubAnalyzer(default_category="pneumothorax")
        self._run(analyzer)
        _image, user, system = analyzer.calls[0]
        assert "pneumothorax" in system
        assert "Pneumothorax" in user

    def test_prompts_still_carry_the_student_text_and_context(self) -> None:
        analyzer = StubAnalyzer(default_category="pneumothorax")
        self._run(analyzer)
        _image, user, system = analyzer.calls[0]
        assert "small left apical pneumothorax" in user
        assert "23yo M, sudden chest pain" in system

    def test_no_unsubstituted_placeholders_remain(self) -> None:
        analyzer = StubAnalyzer(default_category="pneumothorax")
        self._run(analyzer)
        _image, user, system = analyzer.calls[0]
        assert "$ground_truth" not in system + user
        assert "$clinical_context" not in system + user

    def test_matching_category_is_left_untouched(self) -> None:
        feedback = self._run(StubAnalyzer(default_category="pneumothorax"))
        assert feedback.raw["category"] == "pneumothorax"
        assert "ground_truth_override" not in feedback.raw

    def test_a_disobedient_category_is_overridden(self) -> None:
        # The model ignored the category it was given. The student must not be
        # shown "Cardiomegaly" on a confirmed pneumothorax.
        feedback = self._run(StubAnalyzer(default_category="cardiomegaly"))
        assert feedback.likely_finding.value == "Pneumothorax"
        assert feedback.raw["category"] == "pneumothorax"

    def test_the_override_records_what_the_model_actually_said(self) -> None:
        # Silently correcting would hide prompt drift; the eval reads this.
        feedback = self._run(StubAnalyzer(default_category="cardiomegaly"))
        assert feedback.raw["ground_truth_override"] is True
        assert feedback.raw["model_category"] == "cardiomegaly"


def _multi_case() -> Case:
    """A case labelled Cardiomegaly|Effusion, reduced to effusion as primary.

    This shape is a third of the real test split, and it is where withholding
    the other confirmed labels does damage: a student who names the
    cardiomegaly is right about the film.
    """
    return Case(
        id="c1",
        image_url="/images/c1.png",
        ground_truth_finding=Finding(
            category=FindingCategory.PLEURAL_EFFUSION,
            label=FindingLabel("Pleural effusion"),
        ),
        secondary_findings=(
            Finding(category=FindingCategory.CARDIOMEGALY, label=FindingLabel("Cardiomegaly")),
        ),
        metadata={"clinical_context": "68yo F, dyspnoea"},
    )


class TestSecondaryFindingsReachTheModel:
    def _run(self, case: Case, analyzer: StubAnalyzer) -> Feedback:
        use_case = _build(analyzer=analyzer, case_repo=_StubCaseRepo(case))
        return use_case.execute(
            SubmitAttemptRequest(case_id="c1", student_id="alice", free_text="Looks like fluid.")
        ).feedback

    def test_a_confirmed_second_finding_is_named_in_the_prompt(self) -> None:
        analyzer = StubAnalyzer(default_category="pleural_effusion")
        self._run(_multi_case(), analyzer)
        _image, _user, system = analyzer.calls[0]
        assert "Also confirmed on this film" in system
        assert "Cardiomegaly" in system

    def test_a_single_finding_case_gets_no_additional_findings_line(self) -> None:
        # An empty line rather than "(none)": nothing for the model to reason
        # about the absence of.
        analyzer = StubAnalyzer(default_category="pneumothorax")
        self._run(_case(), analyzer)
        _image, _user, system = analyzer.calls[0]
        assert "Also confirmed on this film" not in system
        assert "$secondary_findings_line" not in system

    def test_answering_with_a_secondary_finding_is_not_an_override(self) -> None:
        # The model was shown the cardiomegaly, so naming it is compliance.
        feedback = self._run(_multi_case(), StubAnalyzer(default_category="cardiomegaly"))
        assert "ground_truth_override" not in feedback.raw
        assert feedback.raw["category"] == "cardiomegaly"

    def test_an_unconfirmed_category_is_still_overridden(self) -> None:
        feedback = self._run(_multi_case(), StubAnalyzer(default_category="fracture"))
        assert feedback.raw["ground_truth_override"] is True
        assert feedback.raw["model_category"] == "fracture"
        assert feedback.likely_finding.value == "Pleural effusion"


class TestCaseConfirmedCategories:
    def test_primary_comes_first(self) -> None:
        assert _multi_case().confirmed_categories == ("pleural_effusion", "cardiomegaly")

    def test_a_single_finding_case_lists_only_its_primary(self) -> None:
        assert _case().confirmed_categories == ("pneumothorax",)

    def test_a_secondary_repeating_the_primary_is_rejected(self) -> None:
        # Otherwise the prompt would announce the same finding twice, once as
        # the answer and once as an addition to it.
        with pytest.raises(ValidationError):
            Case(
                id="c1",
                image_url="/images/c1.png",
                ground_truth_finding=Finding(
                    category=FindingCategory.CARDIOMEGALY, label=FindingLabel("Cardiomegaly")
                ),
                secondary_findings=(
                    Finding(
                        category=FindingCategory.CARDIOMEGALY, label=FindingLabel("Big heart")
                    ),
                ),
            )
