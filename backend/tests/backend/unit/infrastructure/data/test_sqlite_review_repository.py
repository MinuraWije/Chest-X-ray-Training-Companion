"""Tests for SqliteReviewRepository."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from cxt.domain.entities.review import EducatorReview
from cxt.infrastructure.config.settings import Settings
from cxt.infrastructure.data.database import (
    build_engine,
    build_session_factory,
    init_database,
)
from cxt.infrastructure.data.sqlite_review_repository import SqliteReviewRepository
from cxt.shared.errors import NotFoundError


@pytest.fixture()
def reviews(tmp_path: Path) -> Iterator[SqliteReviewRepository]:
    settings = Settings(database_url=f"sqlite+pysqlite:///{tmp_path / 'cxt.db'}")
    engine = build_engine(settings)
    init_database(engine)
    factory = build_session_factory(engine)
    yield SqliteReviewRepository(factory)


def _new_review(id_: str = "r-1", attempt_id: str = "a-1") -> EducatorReview:
    return EducatorReview.new(
        id=id_,
        attempt_id=attempt_id,
        educator_id="u-edu",
        scores={"finding_recognition": 4, "clarity": 5},
        comment="Good.",
    )


class TestAddAndGet:
    def test_round_trip(self, reviews: SqliteReviewRepository) -> None:
        reviews.add(_new_review())
        r = reviews.get("r-1")
        assert r.scores == {"finding_recognition": 4, "clarity": 5}
        assert r.comment == "Good."

    def test_get_unknown_raises(self, reviews: SqliteReviewRepository) -> None:
        with pytest.raises(NotFoundError):
            reviews.get("nope")


class TestListForAttempt:
    def test_filters(self, reviews: SqliteReviewRepository) -> None:
        reviews.add(_new_review("r-1", "a-1"))
        reviews.add(_new_review("r-2", "a-2"))
        results = reviews.list_for_attempt("a-1")
        assert [r.id for r in results] == ["r-1"]

    def test_empty(self, reviews: SqliteReviewRepository) -> None:
        assert reviews.list_for_attempt("nope") == []
