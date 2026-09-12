from __future__ import annotations

from cxt.domain.entities.feedback import RetrievedPassage
from cxt.domain.ports.attempt_repository import IAttemptRepository
from cxt.domain.ports.auth_service import IAuthService
from cxt.domain.ports.case_repository import ICaseRepository
from cxt.domain.ports.feedback_generator import IFeedbackGenerator
from cxt.domain.ports.image_analyzer import IImageAnalyzer
from cxt.domain.ports.medgemma_client import IMedGemmaClient
from cxt.domain.ports.retriever import IRetriever
from cxt.domain.ports.review_repository import IReviewRepository
from cxt.domain.ports.user_repository import IUserRepository


def test_ports_are_protocols() -> None:
    """Each port should be a Protocol with the documented methods."""
    for proto in (
        IMedGemmaClient,
        IImageAnalyzer,
        IRetriever,
        IFeedbackGenerator,
        ICaseRepository,
        IAttemptRepository,
        IUserRepository,
        IReviewRepository,
        IAuthService,
    ):
        assert hasattr(proto, "__protocol_attrs__") or hasattr(proto, "_is_protocol"), (
            f"{proto.__name__} should be a typing.Protocol"
        )


def test_retrieved_passage_is_value_object() -> None:
    p = RetrievedPassage(text="x", source="y", score=0.5)
    assert p.score == 0.5
