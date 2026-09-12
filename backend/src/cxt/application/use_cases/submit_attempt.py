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
import re
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from cxt.domain.entities.case import Case
from cxt.domain.entities.feedback import Feedback, RetrievedPassage
from cxt.domain.entities.student_attempt import StudentAttempt
from cxt.domain.ports.attempt_repository import IAttemptRepository
from cxt.domain.ports.case_repository import ICaseRepository
from cxt.domain.ports.image_analyzer import IImageAnalyzer
from cxt.domain.ports.image_loader import IImageLoader
from cxt.domain.ports.retriever import IRetriever, RetrievalQuery
from cxt.domain.value_objects.rationale import Rationale
from cxt.infrastructure.ml.image_loader import ByReferenceImageLoader
from cxt.infrastructure.ml.output_parser import OutputParser
from cxt.infrastructure.ml.prompts import PromptBuilder, load_default_prompts
from cxt.infrastructure.ml.resilient_analyzer import RetryingFeedbackParser
from cxt.infrastructure.retrieval.misconception import named_categories
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
        retrieval_mode: str = "null",
        retrieval_k: int = 3,
        corpus_version: str = "",
    ) -> None:
        self._analyzer = analyzer
        self._attempts = attempt_repository
        self._cases = case_repository
        self._retriever = retriever
        # Labels, not behaviour. The use case never branches on the mode — which
        # arm is running is a composition-root choice — but every feedback row
        # has to say which one produced it or the arms cannot be compared.
        self._retrieval_mode = retrieval_mode
        self._retrieval_k = retrieval_k
        self._corpus_version = corpus_version
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
        # Retrieval happens BEFORE generation. In Phase 1 it ran afterwards and
        # the passages were stapled on, which meant they grounded nothing: a
        # grounding metric over a post-hoc attachment measures only whether
        # prose written in ignorance happens to agree with a document it never
        # saw. The whole point of Phase 2 is that the model reads them.
        passages = self._retrieve(case, attempt)
        prompts = self._prompt_builder.build(
            clinical_context=case.metadata.get("clinical_context", ""),
            student_free_text=attempt.free_text,
            # The case record already holds the verified finding. Supplying it
            # keeps the model on the task the student needs — explaining the
            # film and marking their answer — rather than re-deriving a
            # diagnosis it gets wrong most of the time.
            finding=case.ground_truth_finding,
            secondary_findings=case.secondary_findings,
            passages=passages,
        )
        image = self._image_loader.load(case.image_url)
        result = self._analysis.analyze_and_parse(
            attempt_id=attempt.id,
            image=image,
            prompt=prompts.user,
            system=prompts.system,
        )
        feedback = _enforce_ground_truth(result.feedback, case)
        feedback = _enforce_citations(feedback, passages)
        feedback = _with_provenance(
            feedback,
            prompt_version=self._prompt_builder.version,
            model=result.model_output.model,
            repair_attempts=result.repair_attempts,
            passages=passages,
            retrieval_mode=self._retrieval_mode,
            corpus_version=self._corpus_version,
        )
        # `sources` records what was *supplied*, in the order the [Sn] labels
        # were assigned, so a stored row can be replayed against the prompt the
        # model actually saw.
        return replace(feedback, sources=passages)

    def _retrieve(
        self, case: Case, attempt: StudentAttempt
    ) -> tuple[RetrievedPassage, ...]:
        """Fetch evidence, degrading to none rather than failing the submission.

        A broken corpus index must not cost a student their attempt — the
        Phase 1 behaviour (no passages) is a perfectly good answer. The
        consequence is that the eval has to assert sources are non-empty; left
        unasserted, a silently-empty retriever would let a whole arm run,
        measure nothing, and report success.
        """
        query = RetrievalQuery(
            free_text=attempt.free_text,
            confirmed_categories=case.confirmed_categories,
            named_categories=named_categories(attempt.free_text),
            case_id=case.id,
        )
        try:
            return tuple(self._retriever.retrieve(query=query, top_k=self._retrieval_k))
        except Exception as exc:
            logger.warning(
                "Retriever failed for attempt %s, continuing without evidence: %s",
                attempt.id,
                exc,
            )
            return ()


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


_CITATION_RE = re.compile(r"\[S(\d+)\]")


def _enforce_citations(
    feedback: Feedback, passages: tuple[RetrievedPassage, ...]
) -> Feedback:
    """Strip any `[Sn]` naming a passage that was never supplied.

    Mirrors `_enforce_ground_truth`, for the same reason and with the same
    resolution. The model is handed exactly `len(passages)` labels; anything
    beyond that is a citation to a document that does not exist, and a
    fabricated citation is the single worst failure mode for a tool whose
    premise is "the feedback cites published evidence". A student cannot tell
    a real [S4] from an invented one, so it never reaches them.

    The removal is recorded rather than swallowed: `citations_stripped` is the
    confabulation signal the eval reads, and a rising rate means the prompt has
    drifted.
    """
    supplied = len(passages)
    stripped: list[str] = []

    def _scrub(text: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            index = int(match.group(1))
            if 1 <= index <= supplied:
                return match.group(0)
            stripped.append(match.group(0))
            # Removed rather than rewritten to a valid label: silently
            # repointing a fabricated citation at a real passage would assert
            # that passage says something nobody checked.
            return ""
        return _CITATION_RE.sub(_replace, text)

    def _clean(value: Rationale) -> Rationale:
        scrubbed = _scrub(value.value)
        if scrubbed == value.value:
            return value
        # `Rationale` rejects an empty string, so a field that was nothing but
        # a fabricated citation keeps its original text rather than taking the
        # whole submission down over a formatting failure.
        return Rationale(scrubbed.strip() or value.value)

    cleaned = replace(
        feedback,
        key_signs=_clean(feedback.key_signs),
        comparison=_clean(feedback.comparison),
        missed_clues=_clean(feedback.missed_clues),
        learning_point=_clean(feedback.learning_point),
        key_observations=(
            _clean(feedback.key_observations)
            if feedback.key_observations is not None
            else None
        ),
    )
    if not stripped:
        return feedback
    logger.warning(
        "Model cited %s but only %d passage(s) were supplied; removed.",
        ", ".join(sorted(set(stripped))),
        supplied,
    )
    raw = dict(cleaned.raw)
    raw["citations_stripped"] = sorted(set(stripped))
    return replace(cleaned, raw=raw)


def _with_provenance(
    feedback: Feedback,
    *,
    prompt_version: str,
    model: str,
    repair_attempts: int,
    passages: tuple[RetrievedPassage, ...] = (),
    retrieval_mode: str = "null",
    corpus_version: str = "",
) -> Feedback:
    """Record which prompt, model and corpus produced this feedback (NFR-4).

    The retrieval fields are part of provenance, not telemetry: two runs are
    only comparable if the corpus matched, and a stored row you cannot
    regenerate is not provenance.
    """
    raw = dict(feedback.raw)
    raw["prompt_version"] = prompt_version
    raw["model"] = model
    raw["repair_attempts"] = repair_attempts
    raw["retrieval_mode"] = retrieval_mode
    raw["retrieval_k"] = len(passages)
    raw["corpus_version"] = corpus_version
    # An abstention is a deliberate outcome, not a missing value. The eval
    # reports its rate: a system that knows when it has nothing to cite is a
    # better artefact than one with higher precision and no such notion.
    raw["abstained"] = not passages
    raw["retrieval"] = [
        {"id": p.id, "score": float(p.score), "rank": rank}
        for rank, p in enumerate(passages, start=1)
    ]
    return replace(feedback, raw=raw)


def _utcnow() -> datetime:
    return datetime.now(UTC)
