"""IRetriever port: optional evidence retrieval for the feedback.

Phase 1 ships `NullRetriever` (returns `()`). Phase 2 will add
`VectorRetriever` over the curated teaching corpus. Use cases must not
branch on the retriever type (Liskov).
"""
from __future__ import annotations

from typing import Protocol

from cxt.domain.entities.feedback import RetrievedPassage


class IRetriever(Protocol):
    def retrieve(self, *, query: str, top_k: int = 3) -> tuple[RetrievedPassage, ...]: ...
