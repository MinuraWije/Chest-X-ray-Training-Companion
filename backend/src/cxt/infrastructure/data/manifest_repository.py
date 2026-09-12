"""CSV-backed Case repository for the curated NIH subset.

This is the Phase 1 implementation of ICaseRepository. The Phase 2 swap to a
SQL-backed implementation is a separate adapter behind the same port.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from cxt.domain.entities.case import Case
from cxt.domain.entities.finding import Finding, FindingCategory
from cxt.domain.ports.case_repository import ICaseRepository
from cxt.domain.value_objects.finding_label import FindingLabel
from cxt.infrastructure.data.nih import nih_labels_to_finding_categories
from cxt.shared.errors import NotFoundError, ValidationError

_CATEGORY_LABEL: dict[FindingCategory, str] = {
    FindingCategory.PNEUMOTHORAX: "Pneumothorax",
    FindingCategory.PLEURAL_EFFUSION: "Pleural effusion",
    FindingCategory.CARDIOMEGALY: "Cardiomegaly",
    FindingCategory.CONSOLIDATION: "Consolidation",
    FindingCategory.NODULE_OR_MASS: "Nodule or mass",
    FindingCategory.FRACTURE: "Fracture",
    FindingCategory.NO_FINDING: "No finding",
}


class ManifestRepository(ICaseRepository):
    """Read-only Case repository backed by a manifest.csv file."""

    def __init__(self, manifest_path: Path, images_url_prefix: str = "/images/") -> None:
        if not isinstance(manifest_path, Path):
            raise ValidationError("manifest_path must be a Path.")
        if not manifest_path.exists():
            raise ValidationError(f"Manifest not found: {manifest_path}")
        if not isinstance(images_url_prefix, str) or not images_url_prefix:
            raise ValidationError("images_url_prefix must be a non-empty string.")
        self._path = manifest_path
        self._images_url_prefix = images_url_prefix
        self._rows: list[dict[str, str]] = self._load(manifest_path)

    @staticmethod
    def _load(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [dict(row) for row in reader]

    def _to_case(self, row: dict[str, str]) -> Case:
        finding_value = row["finding"]
        try:
            category = FindingCategory(finding_value)
        except ValueError as exc:
            raise ValidationError(
                f"Invalid finding value in manifest: {finding_value!r}"
            ) from exc
        image_filename = Path(row["image_path"]).name
        image_url = f"{self._images_url_prefix}{image_filename}"
        label_text = row.get("ground_truth_label") or _CATEGORY_LABEL[category]
        nih_labels = [part for part in row.get("nih_labels", "").split("|") if part]
        # `finding` is whatever `find_primary_category` reduced these labels to.
        # The rest are just as confirmed, so they are carried on the case rather
        # than discarded; the prompt hands them to the model so the tutor marks
        # against the whole answer key instead of a third of it.
        secondary = tuple(
            Finding(category=extra, label=FindingLabel(_CATEGORY_LABEL[extra]))
            for extra in nih_labels_to_finding_categories(nih_labels)
            # NO_FINDING is the absence of findings, never an additional one. It
            # can surface here when a row's labels map to none of our six
            # (`Edema|Emphysema`, say), which would otherwise tell the model
            # "also present on this film: no finding".
            if extra != category and extra != FindingCategory.NO_FINDING
        )
        metadata: dict[str, Any] = {
            "label_source": row.get("label_source", "unknown"),
            "nih_labels": nih_labels,
            "image_index": row.get("image_index", ""),
            "patient_age": row.get("patient_age", ""),
            "patient_sex": row.get("patient_sex", ""),
            "view_position": row.get("view_position", ""),
        }
        return Case(
            id=row["case_id"],
            image_url=image_url,
            ground_truth_finding=Finding(
                category=category,
                label=FindingLabel(label_text),
            ),
            split=row.get("split", "train"),
            secondary_findings=secondary,
            metadata=metadata,
        )

    def get(self, case_id: str) -> Case:
        for row in self._rows:
            if row.get("case_id") == case_id:
                return self._to_case(row)
        raise NotFoundError(f"Case not found: {case_id}")

    def list(
        self,
        *,
        split: str | None = None,
        finding: FindingCategory | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Case]:
        filtered: list[dict[str, str]] = []
        for row in self._rows:
            if split is not None and row.get("split") != split:
                continue
            if finding is not None and row.get("finding") != finding.value:
                continue
            filtered.append(row)
        page = filtered[offset:] if offset else filtered
        if limit is not None:
            page = page[:limit]
        return [self._to_case(r) for r in page]

    def count(
        self,
        *,
        split: str | None = None,
        finding: FindingCategory | None = None,
    ) -> int:
        n = 0
        for row in self._rows:
            if split is not None and row.get("split") != split:
                continue
            if finding is not None and row.get("finding") != finding.value:
                continue
            n += 1
        return n
