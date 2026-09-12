"""NullRetriever — IRetriever that returns no passages.

Phase 1 has no RAG corpus. The retriever is a port the use case
calls, so swapping in a real vector retriever in Phase 2 is a
single-line change in the use case factory.
"""
from __future__ import annotations

from dataclasses import dataclass

from cxt.domain.entities.feedback import RetrievedPassage


@dataclass(frozen=True, slots=True)
class NullRetriever:
    def retrieve(self, *, query: str, top_k: int = 3) -> tuple[RetrievedPassage, ...]:
        return ()
