"""IRetriever port: evidence retrieval for the feedback.

Phase 1 shipped `NullRetriever` (returns `()`). Phase 2 adds BM25, oracle and
misconception-keyed implementations behind the same port. Use cases must not
branch on the retriever type (Liskov) — which arm is running is a composition
-root decision, recorded in `Feedback.raw`, never an `if` in the use case.

`retrieve` takes a `RetrievalQuery` struct rather than a widening keyword list.
Two of the three strategies already need signals the others ignore — the
oracle needs the case's confirmed categories, the misconception retriever needs
the categories the *student* named — and a `query: str` signature would have to
grow a parameter per strategy, changing every implementation each time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from cxt.domain.entities.feedback import RetrievedPassage
from cxt.shared.errors import ValidationError


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """Everything any retrieval strategy might key on, for one attempt."""

    #: The student's own words. The only input the BM25 arm uses, and the
    #: entire input to misconception detection.
    free_text: str = ""
    #: Categories the case record confirms. The oracle arm's filter; an upper
    #: bound on retrieval, never a retrieval result.
    confirmed_categories: tuple[str, ...] = field(default_factory=tuple)
    #: Categories the student's text appears to name. Feeds the misconception
    #: boost: a category named here but absent from `confirmed_categories` is
    #: what the student got wrong.
    named_categories: tuple[str, ...] = field(default_factory=tuple)
    #: For logging and per-case eval joins only. Never a ranking signal.
    case_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.free_text, str):
            raise ValidationError("RetrievalQuery.free_text must be a string.")
        for name in ("confirmed_categories", "named_categories"):
            if not isinstance(getattr(self, name), tuple):
                raise ValidationError(f"RetrievalQuery.{name} must be a tuple of str.")

    @property
    def misconceived_categories(self) -> tuple[str, ...]:
        """Named by the student but not confirmed by the case — the error."""
        confirmed = set(self.confirmed_categories)
        return tuple(c for c in self.named_categories if c not in confirmed)


class IRetriever(Protocol):
    def retrieve(
        self, *, query: RetrievalQuery, top_k: int = 3
    ) -> tuple[RetrievedPassage, ...]: ...
