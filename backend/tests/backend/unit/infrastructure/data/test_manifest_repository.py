from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import pytest

from cxt.domain.entities.case import Case
from cxt.domain.entities.finding import FindingCategory
from cxt.infrastructure.data.manifest_repository import ManifestRepository
from cxt.shared.errors import NotFoundError, ValidationError


def _write_manifest(
    path: Path,
    rows: Iterable[dict[str, str]],
    columns: Iterable[str] = (
        "case_id",
        "image_index",
        "image_path",
        "finding",
        "label_source",
        "split",
        "nih_labels",
    ),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture()
def manifest_path(tmp_path: Path) -> Path:
    rows = [
        {
            "case_id": "case_aaaa1111",
            "image_index": "00000001_000",
            "image_path": "images/00000001_000.png",
            "finding": "pneumothorax",
            "label_source": "stub",
            "split": "train",
            "nih_labels": "Pneumothorax",
        },
        {
            "case_id": "case_bbbb2222",
            "image_index": "00000002_000",
            "image_path": "images/00000002_000.png",
            "finding": "pleural_effusion",
            "label_source": "stub",
            "split": "val",
            "nih_labels": "Effusion",
        },
        {
            "case_id": "case_cccc3333",
            "image_index": "00000003_000",
            "image_path": "images/00000003_000.png",
            "finding": "no_finding",
            "label_source": "stub",
            "split": "test",
            "nih_labels": "No Finding",
        },
        {
            "case_id": "case_dddd4444",
            "image_index": "00000004_000",
            "image_path": "images/00000004_000.png",
            "finding": "consolidation",
            "label_source": "stub",
            "split": "train",
            "nih_labels": "Infiltration|Pneumonia",
        },
    ]
    path = tmp_path / "manifest.csv"
    _write_manifest(path, rows)
    return path


class TestManifestRepositoryInit:
    def test_rejects_non_path_argument(self) -> None:
        with pytest.raises(ValidationError):
            ManifestRepository("/tmp/x.csv")  # type: ignore[arg-type]

    def test_rejects_missing_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            ManifestRepository(tmp_path / "missing.csv")

    def test_accepts_existing_manifest(self, manifest_path: Path) -> None:
        repo = ManifestRepository(manifest_path)
        assert repo.count() == 4


class TestManifestRepositoryGet:
    def test_returns_correct_case(self, manifest_path: Path) -> None:
        repo = ManifestRepository(manifest_path)
        case = repo.get("case_bbbb2222")
        assert isinstance(case, Case)
        assert case.id == "case_bbbb2222"
        assert case.image_url == "/images/00000002_000.png"
        assert case.ground_truth_finding.category == FindingCategory.PLEURAL_EFFUSION
        assert case.ground_truth_finding.label.value == "Pleural effusion"
        assert case.split == "val"

    def test_raises_not_found(self, manifest_path: Path) -> None:
        repo = ManifestRepository(manifest_path)
        with pytest.raises(NotFoundError):
            repo.get("case_doesnotexist")

    def test_metadata_is_preserved(self, manifest_path: Path) -> None:
        repo = ManifestRepository(manifest_path)
        case = repo.get("case_cccc3333")
        assert case.metadata["label_source"] == "stub"
        assert case.metadata["nih_labels"] == ["No Finding"]
        assert case.metadata["image_index"] == "00000003_000"

    def test_consolidation_label_uses_canonical_text(self, manifest_path: Path) -> None:
        repo = ManifestRepository(manifest_path)
        case = repo.get("case_dddd4444")
        assert case.ground_truth_finding.category == FindingCategory.CONSOLIDATION
        assert case.metadata["nih_labels"] == ["Infiltration", "Pneumonia"]


class TestManifestRepositoryList:
    def test_returns_all_when_no_filters(self, manifest_path: Path) -> None:
        repo = ManifestRepository(manifest_path)
        cases = repo.list()
        assert len(cases) == 4

    def test_filters_by_split(self, manifest_path: Path) -> None:
        repo = ManifestRepository(manifest_path)
        train = repo.list(split="train")
        assert {c.id for c in train} == {"case_aaaa1111", "case_dddd4444"}

    def test_filters_by_finding(self, manifest_path: Path) -> None:
        repo = ManifestRepository(manifest_path)
        cases = repo.list(finding=FindingCategory.NO_FINDING)
        assert len(cases) == 1
        assert cases[0].id == "case_cccc3333"

    def test_limit_and_offset(self, manifest_path: Path) -> None:
        repo = ManifestRepository(manifest_path)
        first_two = repo.list(limit=2, offset=0)
        last_two = repo.list(limit=2, offset=2)
        assert len(first_two) == 2
        assert len(last_two) == 2
        first_ids = {c.id for c in first_two}
        last_ids = {c.id for c in last_two}
        assert first_ids.isdisjoint(last_ids)

    def test_combined_filters(self, manifest_path: Path) -> None:
        repo = ManifestRepository(manifest_path)
        cases = repo.list(split="train", finding=FindingCategory.CONSOLIDATION)
        assert len(cases) == 1
        assert cases[0].id == "case_dddd4444"


class TestManifestRepositoryCount:
    def test_count_all(self, manifest_path: Path) -> None:
        repo = ManifestRepository(manifest_path)
        assert repo.count() == 4

    def test_count_by_finding(self, manifest_path: Path) -> None:
        repo = ManifestRepository(manifest_path)
        assert repo.count(finding=FindingCategory.PNEUMOTHORAX) == 1
        assert repo.count(finding=FindingCategory.NO_FINDING) == 1

    def test_count_by_split(self, manifest_path: Path) -> None:
        repo = ManifestRepository(manifest_path)
        assert repo.count(split="train") == 2
        assert repo.count(split="test") == 1


class TestManifestRepositoryBadData:
    def test_invalid_finding_value_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        _write_manifest(
            path,
            [
                {
                    "case_id": "x",
                    "image_index": "i",
                    "image_path": "images/i.png",
                    "finding": "not-a-real-category",
                    "label_source": "stub",
                    "split": "train",
                    "nih_labels": "No Finding",
                }
            ],
        )
        repo = ManifestRepository(path)
        with pytest.raises(ValidationError):
            repo.get("x")


class TestSecondaryFindings:
    """The manifest's `finding` column is a reduction; the rest is recoverable.

    `find_primary_category` keeps the highest-priority category and drops the
    others, but `nih_labels` still holds them. A third of the real test split
    carries more than one of our six, so what this class pins is whether a case
    record describes the whole film or only part of it.
    """

    def _repo(self, tmp_path: Path, nih_labels: str, finding: str) -> ManifestRepository:
        path = tmp_path / "m.csv"
        _write_manifest(
            path,
            [
                {
                    "case_id": "case_multi",
                    "image_index": "00000001_002",
                    "image_path": "images/00000001_002.png",
                    "finding": finding,
                    "label_source": "stub",
                    "split": "test",
                    "nih_labels": nih_labels,
                }
            ],
        )
        return ManifestRepository(path)

    def test_a_second_category_becomes_a_secondary_finding(self, tmp_path: Path) -> None:
        case = self._repo(tmp_path, "Cardiomegaly|Effusion", "pleural_effusion").get("case_multi")
        assert [f.category.value for f in case.secondary_findings] == ["cardiomegaly"]
        assert case.confirmed_categories == ("pleural_effusion", "cardiomegaly")

    def test_labels_collapsing_to_one_category_yield_no_secondary(self, tmp_path: Path) -> None:
        # Infiltration and Pneumonia both map to consolidation, so this case is
        # multi-label in NIH terms but single-finding in ours.
        case = self._repo(tmp_path, "Infiltration|Pneumonia", "consolidation").get("case_multi")
        assert case.secondary_findings == ()

    def test_a_single_label_case_has_no_secondary(self, tmp_path: Path) -> None:
        case = self._repo(tmp_path, "Pneumothorax", "pneumothorax").get("case_multi")
        assert case.secondary_findings == ()

    def test_a_negative_case_has_no_secondary(self, tmp_path: Path) -> None:
        case = self._repo(tmp_path, "No Finding", "no_finding").get("case_multi")
        assert case.secondary_findings == ()

    def test_unmapped_labels_do_not_add_no_finding_as_a_secondary(self, tmp_path: Path) -> None:
        # Edema and Emphysema are NIH labels outside our six. They must not
        # arrive as "also confirmed: no finding" on a positive case.
        case = self._repo(tmp_path, "Cardiomegaly|Edema", "cardiomegaly").get("case_multi")
        assert case.secondary_findings == ()

    def test_three_categories_are_all_carried(self, tmp_path: Path) -> None:
        case = self._repo(
            tmp_path, "Effusion|Infiltration|Pneumothorax", "pneumothorax"
        ).get("case_multi")
        assert set(case.confirmed_categories) == {
            "pneumothorax",
            "pleural_effusion",
            "consolidation",
        }

    def test_listing_populates_secondaries_too(self, tmp_path: Path) -> None:
        # `list` builds cases by the same path as `get`; the eval and the case
        # browser both read through it.
        cases = self._repo(tmp_path, "Cardiomegaly|Effusion", "pleural_effusion").list(
            split="test"
        )
        assert [f.category.value for f in cases[0].secondary_findings] == ["cardiomegaly"]
