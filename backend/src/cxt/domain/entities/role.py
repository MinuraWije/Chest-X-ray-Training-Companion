"""Role enum for the three user kinds in Phase 1."""
from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    STUDENT = "STUDENT"
    EDUCATOR = "EDUCATOR"
    ADMIN = "ADMIN"

    @classmethod
    def can_review(cls, role: Role) -> bool:
        return role in (Role.EDUCATOR, Role.ADMIN)

    @classmethod
    def can_submit_attempt(cls, role: Role) -> bool:
        return role is Role.STUDENT
