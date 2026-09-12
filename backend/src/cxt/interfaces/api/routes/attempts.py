"""POST /api/v1/attempts and GET /api/v1/attempts/{id}.

Both routes require a JWT bearer token. The student's id is taken from
the verified token; no X-Student-Id header is honoured in Step 5+.
"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Path

from cxt.application.use_cases.submit_attempt import (
    SubmitAttemptRequest,
    SubmitAttemptUseCase,
)
from cxt.domain.entities.user import User
from cxt.domain.ports.attempt_repository import IAttemptRepository
from cxt.domain.ports.case_repository import ICaseRepository
from cxt.domain.ports.image_analyzer import IImageAnalyzer
from cxt.domain.ports.image_loader import IImageLoader
from cxt.domain.ports.retriever import IRetriever
from cxt.infrastructure.config.settings import Settings, get_settings
from cxt.infrastructure.ml.output_parser import OutputParser
from cxt.infrastructure.ml.prompts import PromptBuilder, load_default_prompts
from cxt.interfaces.api.dependencies import (
    get_attempt_repository,
    get_case_repository,
    get_image_analyzer,
    get_image_loader,
    get_retriever,
    require_role,
)
from cxt.shared.errors import NotFoundError, UpstreamUnavailableError, ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["attempts"])

# Mirrors the frontend's Zod bounds on the interpretation textarea.
_MIN_FREE_TEXT = 20
_MAX_FREE_TEXT = 2000


def _case_exists(repo: ICaseRepository, case_id: str) -> bool:
    try:
        repo.get(case_id)
        return True
    except NotFoundError:
        return False


def _build_use_case(
    *,
    analyzer: IImageAnalyzer,
    attempts: IAttemptRepository,
    cases: ICaseRepository,
    retriever: IRetriever,
    image_loader: IImageLoader,
    max_repair_attempts: int,
) -> SubmitAttemptUseCase:
    return SubmitAttemptUseCase(
        analyzer=analyzer,
        attempt_repository=attempts,
        case_repository=cases,
        retriever=retriever,
        image_loader=image_loader,
        prompt_builder=PromptBuilder(templates=load_default_prompts()),
        output_parser=OutputParser(),
        max_repair_attempts=max_repair_attempts,
    )


def _feedback_to_dto(feedback) -> dict[str, object]:  # type: ignore[no-untyped-def]
    raw = feedback.raw if isinstance(feedback.raw, dict) else {}
    parsed = raw.get("parsed", {}) if isinstance(raw, dict) else {}
    primary = parsed.get("primary_finding", {}) if isinstance(parsed, dict) else {}
    return {
        "attemptId": feedback.attempt_id,
        "likelyFinding": feedback.likely_finding.value.value,
        "keySigns": feedback.key_signs.value,
        "comparison": feedback.comparison.value,
        "missedClues": feedback.missed_clues.value,
        "learningPoint": feedback.learning_point.value,
        "confidence": float(feedback.confidence.value),
        "keyObservations": (
            feedback.key_observations.value if feedback.key_observations is not None else None
        ),
        "differentials": [
            {"label": d.label, "confidence": float(d.confidence.value)}
            for d in feedback.differentials
        ],
        "sources": [
            {"source": s.source, "text": s.text, "score": float(s.score)}
            for s in feedback.sources
        ],
        "primary": primary,
    }


def _attempt_to_dto(attempt, feedback=None) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return {
        "id": attempt.id,
        "caseId": attempt.case_id,
        "studentId": attempt.student_id,
        "freeText": attempt.free_text,
        "submittedAt": attempt.submitted_at.isoformat(),
        "feedback": _feedback_to_dto(feedback) if feedback is not None else None,
    }


def _shape_feedback(feedback) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """Re-shape the internal Feedback entity into the FE-facing `Feedback` interface."""
    label = feedback.likely_finding.value
    # `likely_finding` is the model's human-readable label ("Right apical
    # pneumothorax"); the canonical category is recorded separately by the parser.
    raw = feedback.raw if isinstance(feedback.raw, dict) else {}
    category = raw.get("category") or label
    primary_dict = {
        "category": category,
        "label": label,
        "confidence": float(feedback.confidence.value),
        "rationale": feedback.key_signs.value,
        "source": "model",
    }
    key_observations = (
        feedback.key_observations.value
        if feedback.key_observations is not None
        else feedback.key_signs.value
    )
    return {
        "primaryFinding": primary_dict,
        "differentialFindings": [
            {
                "category": d.label,
                "label": d.label,
                "confidence": float(d.confidence.value),
                "rationale": "",
                "source": "model",
            }
            for d in feedback.differentials
        ],
        "keyObservations": key_observations,
        "teachingPoint": feedback.learning_point.value,
        "sources": [
            {"source": s.source, "text": s.text, "score": float(s.score)}
            for s in feedback.sources
        ],
        "raw": {},
    }


@router.post("/attempts")
def submit_attempt(
    body: dict[str, object],
    user: User = Depends(require_role("STUDENT")),
    analyzer: IImageAnalyzer = Depends(get_image_analyzer),
    attempts: IAttemptRepository = Depends(get_attempt_repository),
    cases: ICaseRepository | None = Depends(get_case_repository),
    retriever: IRetriever = Depends(get_retriever),
    image_loader: IImageLoader = Depends(get_image_loader),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    if not isinstance(body, dict):
        raise ValidationError("Request body must be a JSON object.")

    case_id = body.get("caseId")
    free_text = body.get("freeText")
    if not isinstance(case_id, str) or not case_id:
        raise ValidationError("'caseId' is required.")
    if not isinstance(free_text, str) or not free_text.strip():
        raise ValidationError("'freeText' is required and must be non-empty.")
    # Matches the frontend's Zod bounds. Rejecting here gives a field-level
    # message rather than a confusing 422 from the model runtime.
    if not _MIN_FREE_TEXT <= len(free_text.strip()) <= _MAX_FREE_TEXT:
        raise ValidationError(
            f"'freeText' must be between {_MIN_FREE_TEXT} and {_MAX_FREE_TEXT} characters "
            f"(got {len(free_text.strip())})."
        )

    if cases is None:
        raise UpstreamUnavailableError(
            "No curated case manifest found. From the backend folder run: "
            "python scripts\\dev.py real-data"
        )

    if not _case_exists(cases, case_id):
        raise NotFoundError(f"Case not found: {case_id}")

    use_case = _build_use_case(
        analyzer=analyzer,
        attempts=attempts,
        cases=cases,
        retriever=retriever,
        image_loader=image_loader,
        max_repair_attempts=settings.medgemma_max_repair_attempts,
    )
    try:
        result = use_case.execute(
            SubmitAttemptRequest(
                case_id=case_id,
                student_id=user.id,
                free_text=free_text,
            )
        )
    except (NotFoundError, ValidationError, UpstreamUnavailableError):
        raise
    except Exception as exc:
        logger.exception("submit_attempt failed")
        raise UpstreamUnavailableError(f"submit_attempt failed: {exc}") from exc

    return {
        **_attempt_to_dto(result.attempt),
        "feedback": _shape_feedback(result.feedback),
    }


@router.get("/attempts/{attempt_id}")
def get_attempt(
    attempt_id: Annotated[str, Path(min_length=1, max_length=64)],
    user: User = Depends(require_role("STUDENT", "EDUCATOR", "ADMIN")),
    attempts: IAttemptRepository = Depends(get_attempt_repository),
) -> dict[str, object]:
    attempt, feedback = attempts.get_with_feedback(attempt_id)
    if user.role.value == "STUDENT" and user.id != attempt.student_id:
        raise NotFoundError(f"Attempt not found: {attempt_id}")
    return {
        **_attempt_to_dto(attempt),
        "feedback": _shape_feedback(feedback) if feedback is not None else None,
    }


@router.get("/attempts")
def list_my_attempts(
    user: User = Depends(require_role("STUDENT", "EDUCATOR", "ADMIN")),
    attempts: IAttemptRepository = Depends(get_attempt_repository),
) -> dict[str, object]:
    if user.role.value == "STUDENT":
        rows = attempts.list_for_student(user.id)
    else:
        rows = attempts.list_for_educator_review()
    return {
        "items": [
            {
                **_attempt_to_dto(attempt),
                "feedback": None,
            }
            for attempt in rows
        ],
        "total": len(rows),
    }
