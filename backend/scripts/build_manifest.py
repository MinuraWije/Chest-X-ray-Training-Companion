"""Build a curated case manifest from data/raw/ into data/processed/.

Reads PROVENANCE.json (written by download_nih_subset) plus the optional NIH
Data_Entry_2017.csv, applies the NIH -> our-6-finding mapping, balances the
subset, and writes:

    data/processed/manifest.csv
    data/processed/case_index.json

manifest.csv columns:
    case_id, image_index, image_path, finding, label_source, split, nih_labels

The "split" column deterministically assigns each case to {train, val, test}
in a 70/15/15 ratio using a hash of the case_id so reruns are reproducible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cxt.domain.entities.finding import FindingCategory
from cxt.infrastructure.data.nih import NO_FINDING_LABEL, find_primary_category

logger = logging.getLogger("cxt.scripts.build_manifest")


REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
RAW_DIR_DEFAULT = REPO_ROOT_DEFAULT / "data" / "raw"
PROCESSED_DIR_DEFAULT = REPO_ROOT_DEFAULT / "data" / "processed"
IMAGES_DIR_NAME = "images"
PROVENANCE_FILENAME = "PROVENANCE.json"
METADATA_FILENAME = "Data_Entry_2017.csv"
MANIFEST_FILENAME = "manifest.csv"
INDEX_FILENAME = "case_index.json"

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


@dataclass(frozen=True, slots=True)
class CaseRow:
    case_id: str
    image_index: str
    image_path: str
    finding: FindingCategory
    label_source: str
    split: str
    nih_labels: tuple[str, ...]
    patient_age: str = ""
    patient_sex: str = ""
    view_position: str = ""

    def to_csv(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "image_index": self.image_index,
            "image_path": self.image_path,
            "finding": self.finding.value,
            "label_source": self.label_source,
            "split": self.split,
            "nih_labels": "|".join(self.nih_labels),
            "patient_age": self.patient_age,
            "patient_sex": self.patient_sex,
            "view_position": self.view_position,
        }


def _deterministic_split(case_id: str) -> str:
    h = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest(), 16)
    bucket = (h % 1000) / 1000.0
    acc = 0.0
    for split, ratio in SPLIT_RATIOS.items():
        acc += ratio
        if bucket < acc:
            return split
    return "test"


def _load_provenance(raw_dir: Path) -> list[dict[str, object]]:
    path = raw_dir / PROVENANCE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Provenance not found at {path}. Run download_nih_subset.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("PROVENANCE.json missing 'files' array.")
    return [dict(entry) for entry in files]


def _balance(
    rows: list[CaseRow], n_per_finding: int, n_negatives: int
) -> list[CaseRow]:
    by_finding: dict[FindingCategory, list[CaseRow]] = {}
    for row in rows:
        by_finding.setdefault(row.finding, []).append(row)
    selected: list[CaseRow] = []
    for category in (
        FindingCategory.PNEUMOTHORAX,
        FindingCategory.PLEURAL_EFFUSION,
        FindingCategory.CARDIOMEGALY,
        FindingCategory.CONSOLIDATION,
        FindingCategory.NODULE_OR_MASS,
        FindingCategory.FRACTURE,
    ):
        bucket = by_finding.get(category, [])
        if len(bucket) < n_per_finding:
            logger.warning(
                "Only %d cases available for %s (wanted %d).",
                len(bucket),
                category.value,
                n_per_finding,
            )
        selected.extend(bucket[:n_per_finding])
    negatives = by_finding.get(FindingCategory.NO_FINDING, [])
    if len(negatives) < n_negatives:
        logger.warning("Only %d negatives available (wanted %d).", len(negatives), n_negatives)
    selected.extend(negatives[:n_negatives])
    return selected


def _stable_case_id(image_index: str, primary: str) -> str:
    h = hashlib.sha1(f"{image_index}|{primary}".encode()).hexdigest()[:8]
    return f"case_{h}"


def _copy_images(rows: Iterable[CaseRow], raw_images: Path, processed_images: Path) -> None:
    processed_images.mkdir(parents=True, exist_ok=True)
    for row in rows:
        src = raw_images / row.image_index
        if not src.exists():
            candidates = list(raw_images.glob(f"{row.image_index}.*"))
            if not candidates:
                logger.error("Missing raw image for %s; skipping copy.", row.image_index)
                continue
            src = candidates[0]
        dest = processed_images / src.name
        if not dest.exists():
            shutil.copy2(src, dest)


def _write_manifest(rows: list[CaseRow], processed_dir: Path) -> Path:
    path = processed_dir / MANIFEST_FILENAME
    fieldnames = list(rows[0].to_csv().keys()) if rows else [
        "case_id",
        "image_index",
        "image_path",
        "finding",
        "label_source",
        "split",
        "nih_labels",
        "patient_age",
        "patient_sex",
        "view_position",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv())
    return path


def _write_index(rows: list[CaseRow], processed_dir: Path) -> Path:
    by_finding: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for r in rows:
        by_finding[r.finding.value] = by_finding.get(r.finding.value, 0) + 1
        by_split[r.split] = by_split.get(r.split, 0) + 1
    payload: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "case_count": len(rows),
        "by_finding": by_finding,
        "by_split": by_split,
        "case_ids": [r.case_id for r in rows],
    }
    path = processed_dir / INDEX_FILENAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR_DEFAULT)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR_DEFAULT)
    parser.add_argument("--n-per-finding", type=int, default=5, help="Positive cases per finding.")
    parser.add_argument("--n-negatives", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    files = _load_provenance(args.raw_dir)
    raw_images = args.raw_dir / IMAGES_DIR_NAME
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    processed_images = args.processed_dir / IMAGES_DIR_NAME

    rows: list[CaseRow] = []
    for entry in files:
        filename = str(entry.get("filename", ""))
        nih_labels_raw: object = entry.get("nih_labels", NO_FINDING_LABEL)
        if isinstance(nih_labels_raw, str):
            nih_labels: tuple[str, ...] = (
                (NO_FINDING_LABEL,)
                if not nih_labels_raw
                else tuple(nih_labels_raw.split("|")) or (NO_FINDING_LABEL,)
            )
        elif isinstance(nih_labels_raw, list):
            nih_labels = tuple(str(item) for item in nih_labels_raw) or (NO_FINDING_LABEL,)
        else:
            nih_labels = (NO_FINDING_LABEL,)
        label_source = str(entry.get("source", "unknown"))
        primary = find_primary_category(nih_labels)
        image_index = Path(filename).stem
        case_id = _stable_case_id(image_index, primary.value)
        split = _deterministic_split(case_id)
        rows.append(
            CaseRow(
                case_id=case_id,
                image_index=image_index,
                image_path=f"{IMAGES_DIR_NAME}/{filename}",
                finding=primary,
                label_source=label_source,
                split=split,
                nih_labels=nih_labels,
                patient_age=str(entry.get("patient_age", "") or ""),
                patient_sex=str(entry.get("patient_sex", "") or ""),
                view_position=str(entry.get("view_position", "") or ""),
            )
        )

    selected = _balance(rows, args.n_per_finding, args.n_negatives)
    if not selected:
        logger.error("Manifest is empty; check balance settings and provenance data.")
        return 1

    _copy_images(selected, raw_images, processed_images)
    manifest_path = _write_manifest(selected, args.processed_dir)
    index_path = _write_index(selected, args.processed_dir)
    logger.info(
        "Wrote %d cases to %s and %s.",
        len(selected),
        manifest_path,
        index_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
