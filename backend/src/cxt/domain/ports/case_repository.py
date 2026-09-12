"""ICaseRepository port: persistence for Case entities (currently CSV-backed)."""
from __future__ import annotations

from typing import Protocol

from cxt.domain.entities.case import Case
from cxt.domain.entities.finding import FindingCategory


class ICaseRepository(Protocol):
    def get(self, case_id: str) -> Case: ...

    def list(
        self,
        *,
        split: str | None = None,
        finding: FindingCategory | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Case]: ...

    def count(self, *, split: str | None = None, finding: FindingCategory | None = None) -> int: ...
