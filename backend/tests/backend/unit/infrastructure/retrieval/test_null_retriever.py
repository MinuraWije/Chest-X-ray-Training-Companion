from __future__ import annotations

from cxt.infrastructure.retrieval.null_retriever import NullRetriever


def test_returns_empty_tuple() -> None:
    retriever = NullRetriever()
    assert retriever.retrieve(query="anything") == ()
    assert retriever.retrieve(query="x", top_k=10) == ()
