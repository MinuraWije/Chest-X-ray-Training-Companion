"""SubmitAttemptUseCase — the orchestration of one student attempt.

Steps:
1. Build prompts from the case + the student's free text.
2. Hand the case image + prompts to the `IImageAnalyzer`.
3. Parse the analyzer's raw text via the `OutputParser` into a
   `Feedback` entity.
4. Attach any retrieved passages (empty in Phase 1).
5. Persist the attempt via `IAttemptRepository`.

Use cases depend only on ports; swapping the MedGemma runtime or the
retriever is a one-line factory change.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from cxt.domain.entities.case import Case
from cxt.domain.entities.feedback import Feedback
from cxt.domain.entities.student_attempt import StudentAttempt
from cxt.domain.ports.attempt_repository import IAttemptRepository
from cxt.domain.ports.case_repository import ICaseRepository
from cxt.domain.ports.image_analyzer import IImageAnalyzer
from cxt.domain.ports.image_loader import IImageLoader
from cxt.domain.ports.retriever import IRetriever
from cxt.infrastructure.ml.image_loader import ByReferenceImageLoader
from cxt.infrastructure.ml.output_parser import OutputParser
from cxt.infrastructure.ml.prompts import PromptBuilder, load_default_prompts
from cxt.infrastructure.ml.resilient_analyzer import RetryingFeedbackParser
from cxt.shared.errors import NotFoundError, UpstreamUnavailableError, ValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SubmitAttemptRequest:
    case_id: str
    student_id: str
    free_text: str


@dataclass(frozen=True, slots=True)
class SubmitAttemptResult:
    attempt: StudentAttempt
    feedback: Feedback


class SubmitAttemptUseCase:
    def __init__(
        self,
        *,
        analyzer: IImageAnalyzer,
        attempt_repository: IAttemptRepository,
        case_repository: ICaseRepository,
        retriever: IRetriever,
        image_loader: IImageLoader | None = None,
        prompt_builder: PromptBuilder | None = None,
        output_parser: OutputParser | None = None,
        max_repair_attempts: int = 1,
    ) -> None:
        self._analyzer = analyzer
        self._attempts = attempt_repository
        self._cases = case_repository
        self._retriever = retriever
        self._image_loader = image_loader or ByReferenceImageLoader()
        self._prompt_builder = prompt_builder or PromptBuilder(templates=load_default_prompts())
        self._output_parser = output_parser or OutputParser()
        self._analysis = RetryingFeedbackParser(
            analyzer,
            self._output_parser,
            max_repair_attempts=max_repair_attempts,
        )

    def execute(self, request: SubmitAttemptRequest) -> SubmitAttemptResult:
        case = self._cases.get(request.case_id)
        if not request.student_id:
            raise ValidationError("SubmitAttemptRequest.student_id is required.")
        if not request.free_text or not request.free_text.strip():
            raise ValidationError("SubmitAttemptRequest.free_text must be non-empty.")

        attempt = StudentAttempt.new(
            id=str(uuid.uuid4()),
            case_id=case.id,
            student_id=request.student_id,
            free_text=request.free_text,
        )

        try:
            self._attempts.add(attempt)
        except ValidationError as exc:
            raise ValidationError(f"Could not persist attempt: {exc}") from exc

        try:
            feedback = self._generate_feedback(case, attempt)
        except (NotFoundError, UpstreamUnavailableError):
            raise
        except Exception as exc:
            logger.exception("Feedback generation failed for attempt %s", attempt.id)
            raise UpstreamUnavailableError(f"Feedback generation failed: {exc}") from exc

        try:
            self._attempts.attach_feedback(attempt.id, feedback)
        except Exception as exc:
            logger.warning("Failed to persist feedback for attempt %s: %s", attempt.id, exc)

        return SubmitAttemptResult(attempt=attempt, feedback=feedback)

    def _generate_feedback(self, case: Case, attempt: StudentAttempt) -> Feedback:
        prompts = self._prompt_builder.build(
            clinical_context=case.metadata.get("clinical_context", ""),
            student_free_text=attempt.free_text,
            # The case record already holds the verified finding. Supplying it
            # keeps the model on the task the student needs — explaining the
            # film and marking their answer — rather than re-deriving a
            # diagnosis it gets wrong most of the time.
            finding=case.ground_truth_finding,
            secondary_findings=case.secondary_findings,
        )
        image = self._image_loader.load(case.image_url)
        result = self._analysis.analyze_and_parse(
            attempt_id=attempt.id,
            image=image,
            prompt=prompts.user,
            system=prompts.system,
        )
        feedback = _enforce_ground_truth(result.feedback, case)
        feedback = _with_provenance(
            feedback,
            prompt_version=self._prompt_builder.version,
            model=result.model_output.model,
            repair_attempts=result.repair_attempts,
        )
        return self._attach_retrieval(feedback, attempt)

    def _attach_retrieval(self, feedback: Feedback, attempt: StudentAttempt) -> Feedback:
        try:
            passages = self._retriever.retrieve(query=attempt.free_text, top_k=3)
        except Exception as exc:
            logger.warning("Retriever failed (ignored): %s", exc)
            return feedback
        if not passages:
            return feedback
        return replace(feedback, sources=tuple(passages))


def _enforce_ground_truth(feedback: Feedback, case: Case) -> Feedback:
    """Guarantee the student is never shown a finding the case does not confirm.

    The prompt hands the model its category and tells it to echo it back, so a
    mismatch is the model overriding an instruction rather than a diagnosis we
    are entitled to weigh. Correcting the displayed label is the safe
    resolution: the surrounding prose may still be off, but a student reading
    "Pneumothorax" on a confirmed effusion is a teaching failure of a different
    order. The mismatch is recorded rather than swallowed — it is the
    compliance signal the eval reads, and a rising rate means the prompt has
    drifted.

    Any *confirmed* category passes, not only the primary. On a case labelled
    `Cardiomegaly|Effusion` the model is shown both, so answering with the
    cardiomegaly is following the prompt, not defying it. Treating that as an
    override would log a compliance failure for correct behaviour and would
    rewrite a true finding into a different true one.
    """
    reported = feedback.raw.get("category")
    if reported in case.confirmed_categories:
        return feedback
    truth = case.ground_truth_finding
    logger.warning(
        "Model returned category %r for a case confirming %s; overriding the "
        "displayed label.",
        reported,
        ", ".join(case.confirmed_categories),
    )
    raw = dict(feedback.raw)
    raw["category"] = truth.category.value
    raw["model_category"] = reported
    raw["ground_truth_override"] = True
    return replace(feedback, likely_finding=truth.label, raw=raw)


def _with_provenance(
    feedback: Feedback,
    *,
    prompt_version: str,
    model: str,
    repair_attempts: int,
) -> Feedback:
    """Record which prompt and model produced this feedback (NFR-4)."""
    raw = dict(feedback.raw)
    raw["prompt_version"] = prompt_version
    raw["model"] = model
    raw["repair_attempts"] = repair_attempts
    return replace(feedback, raw=raw)


def _utcnow() -> datetime:
    return datetime.now(UTC)
