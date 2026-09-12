"""Tests for the review use cases (list queue, submit, list for attempt)."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from cxt.application.use_cases.reviews import (
    ListReviewQueueUseCase,
    ListReviewsForAttemptUseCase,
    ReviewAttemptRequest,
    ReviewAttemptUseCase,
)
from cxt.domain.entities.role import Role
from cxt.domain.entities.student_attempt import StudentAttempt
from cxt.infrastructure.config.settings import Settings
from cxt.infrastructure.data.database import (
    build_engine,
    build_session_factory,
    init_database,
)
from cxt.infrastructure.data.sqlite_attempt_repository import SqliteAttemptRepository
from cxt.infrastructure.data.sqlite_review_repository import SqliteReviewRepository
from cxt.shared.errors import AuthorizationError, ValidationError


@pytest.fixture()
def session_factory(tmp_path: Path) -> Iterator[object]:
    settings = Settings(database_url=f"sqlite+pysqlite:///{tmp_path / 'cxt.db'}")
    engine = build_engine(settings)
    init_database(engine)
    yield build_session_factory(engine)


@pytest.fixture()
def attempts(session_factory: object) -> SqliteAttemptRepository:
    return SqliteAttemptRepository(session_factory)  # type: ignore[arg-type]


@pytest.fixture()
def reviews(session_factory: object) -> SqliteReviewRepository:
    return SqliteReviewRepository(session_factory)  # type: ignore[arg-type]


def _seed_attempt(attempts: SqliteAttemptRepository) -> StudentAttempt:
    attempt = StudentAttempt.new(
        id="a-1", case_id="c-1", student_id="u-1", free_text="interpretation"
    )
    attempts.add(attempt)
    return attempt


class TestListReviewQueueUseCase:
    def test_returns_attempts_for_educator(
        self, attempts: SqliteAttemptRepository
    ) -> None:
        _seed_attempt(attempts)
        use_case = ListReviewQueueUseCase(attempt_repository=attempts)
        result = use_case.execute(caller_role=Role.EDUCATOR)
        assert len(result) == 1
        assert result[0].id == "a-1"

    def test_admin_can_list(self, attempts: SqliteAttemptRepository) -> None:
        _seed_attempt(attempts)
        use_case = ListReviewQueueUseCase(attempt_repository=attempts)
        assert use_case.execute(caller_role=Role.ADMIN)

    def test_student_blocked(self, attempts: SqliteAttemptRepository) -> None:
        use_case = ListReviewQueueUseCase(attempt_repository=attempts)
        with pytest.raises(AuthorizationError):
            use_case.execute(caller_role=Role.STUDENT)


class TestReviewAttemptUseCase:
    def test_educator_can_submit(
        self,
        attempts: SqliteAttemptRepository,
        reviews: SqliteReviewRepository,
    ) -> None:
        _seed_attempt(attempts)
        use_case = ReviewAttemptUseCase(
            review_repository=reviews, attempt_repository=attempts
        )
        review = use_case.execute(
            ReviewAttemptRequest(
                attempt_id="a-1",
                scores={"finding_recognition": 4, "clarity": 5},
                comment="Good.",
                educator_id="u-edu",
                educator_role=Role.EDUCATOR,
            )
        )
        assert review.attempt_id == "a-1"
        assert review.scores["finding_recognition"] == 4

    def test_student_blocked(
        self,
        attempts: SqliteAttemptRepository,
        reviews: SqliteReviewRepository,
    ) -> None:
        _seed_attempt(attempts)
        use_case = ReviewAttemptUseCase(
            review_repository=reviews, attempt_repository=attempts
        )
        with pytest.raises(AuthorizationError):
            use_case.execute(
                ReviewAttemptRequest(
                    attempt_id="a-1",
                    scores={"finding_recognition": 4},
                    comment="",
                    educator_id="u-1",
                    educator_role=Role.STUDENT,
                )
            )

    def test_empty_scores_rejected(
        self,
        attempts: SqliteAttemptRepository,
        reviews: SqliteReviewRepository,
    ) -> None:
        _seed_attempt(attempts)
        use_case = ReviewAttemptUseCase(
            review_repository=reviews, attempt_repository=attempts
        )
        with pytest.raises(ValidationError):
            use_case.execute(
                ReviewAttemptRequest(
                    attempt_id="a-1",
                    scores={},
                    comment="",
                    educator_id="u-edu",
                    educator_role=Role.EDUCATOR,
                )
            )

    def test_unknown_attempt_rejected(
        self,
        attempts: SqliteAttemptRepository,
        reviews: SqliteReviewRepository,
    ) -> None:
        use_case = ReviewAttemptUseCase(
            review_repository=reviews, attempt_repository=attempts
        )
        from cxt.shared.errors import NotFoundError

        with pytest.raises(NotFoundError):
            use_case.execute(
                ReviewAttemptRequest(
                    attempt_id="nope",
                    scores={"finding_recognition": 4},
                    comment="",
                    educator_id="u-edu",
                    educator_role=Role.EDUCATOR,
                )
            )


class TestListReviewsForAttemptUseCase:
    def test_returns_reviews(
        self,
        attempts: SqliteAttemptRepository,
        reviews: SqliteReviewRepository,
    ) -> None:
        _seed_attempt(attempts)
        review_use_case = ReviewAttemptUseCase(
            review_repository=reviews, attempt_repository=attempts
        )
        review_use_case.execute(
            ReviewAttemptRequest(
                attempt_id="a-1",
                scores={"finding_recognition": 4},
                comment="",
                educator_id="u-edu",
                educator_role=Role.EDUCATOR,
            )
        )
        list_use_case = ListReviewsForAttemptUseCase(review_repository=reviews)
        result = list_use_case.execute(attempt_id="a-1", caller_role=Role.EDUCATOR)
        assert len(result) == 1

    def test_student_blocked(self, reviews: SqliteReviewRepository) -> None:
        list_use_case = ListReviewsForAttemptUseCase(review_repository=reviews)
        with pytest.raises(AuthorizationError):
            list_use_case.execute(attempt_id="a-1", caller_role=Role.STUDENT)
