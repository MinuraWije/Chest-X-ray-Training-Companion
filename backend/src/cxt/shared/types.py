"""Common type aliases used across the backend."""
from __future__ import annotations

from typing import NewType

UserId = NewType("UserId", str)
CaseId = NewType("CaseId", str)
AttemptId = NewType("AttemptId", str)
ReviewId = NewType("ReviewId", str)
