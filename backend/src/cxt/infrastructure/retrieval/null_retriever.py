"""`NullRetriever` lives in `retrievers.py` with the other arms.

Kept as a re-export because the composition root, the probe scripts and
several tests import it from here, and moving it is not worth touching eight
call sites to say the same thing.
"""
from __future__ import annotations

from cxt.infrastructure.retrieval.retrievers import NullRetriever

__all__ = ["NullRetriever"]
