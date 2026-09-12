"""Fetch a small subset of chest X-ray images + NIH metadata into data/raw/.

This script supports three sources so the rest of the system can be developed
end-to-end before (or in parallel with) a real NIH download:

    --source stub       Generates 1024x1024 synthetic PNGs with the target
                        finding name rendered. Use for local dev only; the
                        manifest flags these as label_source='stub' so they
                        can never be confused with real clinical data.
    --source local      Copies images from --local-dir; pairs them with rows
                        of --metadata-csv via filename (column
                        'Image Index' or 'image_index').
    --source url-list   Downloads each URL in --url-file into data/raw/images/.

In all cases the script records a PROVENANCE.json with SHA-256 checksums and
the source URL/path for every file written to data/raw/. Nothing here makes
a network call unless --source is 'url-list'.

Usage:
    python -m scripts.download_nih_subset --source stub --n-per-finding 5
    python -m scripts.download_nih_subset --source local \
        --local-dir path/to/cxr --metadata-csv path/to/Data_Entry_2017.csv
    python -m scripts.download_nih_subset --source url-list \
        --url-file urls.txt --metadata-csv Data_Entry_2017.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from PIL import Image, ImageDraw, ImageFont

from cxt.domain.entities.finding import FindingCategory
from cxt.infrastructure.data.nih import (
    NIH_FOURTEEN,
    NO_FINDING_LABEL,
    find_primary_category,
)

logger = logging.getLogger("cxt.scripts.download_nih")


REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[2]
RAW_DIR_DEFAULT = REPO_ROOT_DEFAULT / "data" / "raw"
IMAGES_DIR_NAME = "images"
METADATA_FILENAME = "Data_Entry_2017.csv"
PROVENANCE_FILENAME = "PROVENANCE.json"

STUB_SIZE = (1024, 1024)
STUB_BG = (24, 24, 32)
STUB_FG = (220, 220, 220)
STUB_FONT_SIZE = 36


@dataclass(frozen=True, slots=True)
class ProvenanceEntry:
    filename: str
    sha256: str
    bytes: int
    source: str
    nih_labels: tuple[str, ...]
    primary_category: str
    patient_age: str = ""
    patient_sex: str = ""
    view_position: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "source": self.source,
            "nih_labels": list(self.nih_labels),
            "primary_category": self.primary_category,
            "patient_age": self.patient_age,
            "patient_sex": self.patient_sex,
            "view_position": self.view_position,
        }


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_finding_labels(cell: str) -> tuple[str, ...]:
    cell = (cell or "").strip()
    if not cell or cell == NO_FINDING_LABEL:
        return (NO_FINDING_LABEL,)
    return tuple(part.strip() for part in cell.split("|") if part.strip())


def _metadata_lookup(row: dict[str, str]) -> dict[str, str]:
    """Pick patient demographics from a NIH metadata row, tolerating the
    v2017 ``Patient Gender`` / ``OriginalImageWidth`` column names and the
    v2020 ``Patient Sex`` / ``OriginalImage[Width,Height]`` renames.
    """
    return {
        "patient_age": (row.get("Patient Age") or "").strip(),
        "patient_sex": (row.get("Patient Sex") or row.get("Patient Gender") or "").strip(),
        "view_position": (row.get("View Position") or "").strip(),
    }


def read_metadata_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def _stub_font() -> ImageFont.ImageFont:
    try:
        return cast(
            "ImageFont.ImageFont",
            ImageFont.truetype("arial.ttf", STUB_FONT_SIZE),
        )
    except OSError:
        return cast("ImageFont.ImageFont", ImageFont.load_default())


def _write_stub_image(path: Path, label_text: str) -> None:
    img = Image.new("RGB", STUB_SIZE, STUB_BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, STUB_SIZE[0] - 40, STUB_SIZE[1] - 40), outline=STUB_FG, width=2)
    font = _stub_font()
    draw.text((60, STUB_SIZE[1] // 2 - 30), f"STUB: {label_text}", fill=STUB_FG, font=font)
    draw.text((60, STUB_SIZE[1] // 2 + 20), "Not clinical data", fill=STUB_FG, font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def _download_to(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _target_filename(image_index: str) -> str:
    return f"{image_index}.png"


def _write_provenance(entries: Iterable[ProvenanceEntry], raw_dir: Path) -> None:
    payload = {
        "dataset": "NIH ChestX-ray8 / NIHCC",
        "downloaded_at": datetime.now(UTC).isoformat(),
        "tool": "scripts.download_nih_subset",
        "schema": {
            "metadata_columns": [
                "Image Index",
                "Finding Labels",
                "Follow-up #",
                "Patient ID",
                "Patient Age",
                "Patient Gender",
                "View Position",
                "OriginalImageWidth",
                "OriginalImageHeight",
                "OriginalImagePixelSpacing_x",
                "OriginalImagePixelSpacing_y",
            ],
            "nih_fourteen_labels": list(NIH_FOURTEEN),
            "no_finding_label": NO_FINDING_LABEL,
        },
        "file_metadata_fields": [
            "patient_age",
            "patient_sex",
            "view_position",
        ],
        "files": [e.to_dict() for e in entries],
    }
    path = raw_dir / PROVENANCE_FILENAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote provenance: %s (%d files)", path, len(payload["files"]))


def _select_stub_indices(n_per_finding: int) -> list[tuple[str, tuple[str, ...]]]:
    """Generate (filename_stem, nih_labels) pairs for stub mode."""
    selection: list[tuple[str, tuple[str, ...]]] = []
    counter = 1
    for category in (
        FindingCategory.PNEUMOTHORAX,
        FindingCategory.PLEURAL_EFFUSION,
        FindingCategory.CARDIOMEGALY,
        FindingCategory.CONSOLIDATION,
        FindingCategory.NODULE_OR_MASS,
        FindingCategory.FRACTURE,
    ):
        nih_label = {
            FindingCategory.PNEUMOTHORAX: "Pneumothorax",
            FindingCategory.PLEURAL_EFFUSION: "Effusion",
            FindingCategory.CARDIOMEGALY: "Cardiomegaly",
            FindingCategory.CONSOLIDATION: "Consolidation",
            FindingCategory.NODULE_OR_MASS: "Nodule",
            FindingCategory.FRACTURE: "Fracture",
        }[category]
        for _ in range(n_per_finding):
            stem = f"stub_{counter:05d}"
            counter += 1
            selection.append((stem, (nih_label,)))
    for _ in range(n_per_finding):
        stem = f"stub_{counter:05d}"
        counter += 1
        selection.append((stem, (NO_FINDING_LABEL,)))
    return selection


def _download_stub(raw_dir: Path, n_per_finding: int) -> list[ProvenanceEntry]:
    images_dir = raw_dir / IMAGES_DIR_NAME
    images_dir.mkdir(parents=True, exist_ok=True)
    entries: list[ProvenanceEntry] = []
    for stem, labels in _select_stub_indices(n_per_finding):
        path = images_dir / _target_filename(stem)
        primary = find_primary_category(labels).value
        _write_stub_image(path, f"{primary} (stub)")
        entries.append(
            ProvenanceEntry(
                filename=path.name,
                sha256=sha256_of(path),
                bytes=path.stat().st_size,
                source="stub",
                nih_labels=labels,
                primary_category=primary,
            )
        )
    return entries


def _download_local(
    raw_dir: Path, local_dir: Path, metadata: list[dict[str, str]]
) -> list[ProvenanceEntry]:
    images_dir = raw_dir / IMAGES_DIR_NAME
    images_dir.mkdir(parents=True, exist_ok=True)
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    for path in local_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            by_name[path.name] = path
            by_stem[path.stem] = path

    def _lookup(image_index: str) -> Path | None:
        if image_index in by_name:
            return by_name[image_index]
        bare = Path(image_index).stem
        return by_stem.get(bare)

    entries: list[ProvenanceEntry] = []
    matched = unmatched = 0
    for row in metadata:
        image_index = row.get("Image Index", "").strip()
        if not image_index:
            unmatched += 1
            continue
        src = _lookup(image_index)
        if src is None:
            unmatched += 1
            continue
        labels = parse_finding_labels(row.get("Finding Labels", ""))
        primary = find_primary_category(labels).value
        demographics = _metadata_lookup(row)
        dest = images_dir / _target_filename(Path(image_index).stem)
        _copy_file(src, dest)
        entries.append(
            ProvenanceEntry(
                filename=dest.name,
                sha256=sha256_of(dest),
                bytes=dest.stat().st_size,
                source=f"local:{src}",
                nih_labels=labels,
                primary_category=primary,
                patient_age=demographics["patient_age"],
                patient_sex=demographics["patient_sex"],
                view_position=demographics["view_position"],
            )
        )
        matched += 1
    logger.info("Local source: matched %d / unmatched %d rows", matched, unmatched)
    return entries


def _download_url_list(
    raw_dir: Path,
    url_file: Path,
    metadata: list[dict[str, str]] | None,
) -> list[ProvenanceEntry]:
    images_dir = raw_dir / IMAGES_DIR_NAME
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_by_index: dict[str, tuple[str, ...]] = {}
    if metadata is not None:
        for row in metadata:
            labels_by_index[row.get("Image Index", "").strip()] = parse_finding_labels(
                row.get("Finding Labels", "")
            )
    entries: list[ProvenanceEntry] = []
    with url_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_index, url = [*line.split(maxsplit=1), ""][:2]
            dest = images_dir / _target_filename(image_index)
            try:
                _download_to(url, dest)
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", url, exc)
                continue
            labels = labels_by_index.get(image_index, (NO_FINDING_LABEL,))
            primary = find_primary_category(labels).value
            entries.append(
                ProvenanceEntry(
                    filename=dest.name,
                    sha256=sha256_of(dest),
                    bytes=dest.stat().st_size,
                    source=url,
                    nih_labels=labels,
                    primary_category=primary,
                )
            )
    return entries


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR_DEFAULT,
        help=f"Where to write raw images and metadata (default: {RAW_DIR_DEFAULT}).",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", choices=("stub", "local", "url-list"))
    source.add_argument("--source-stub", dest="source", action="store_const", const="stub")
    source.add_argument("--source-local", dest="source", action="store_const", const="local")
    source.add_argument("--source-url-list", dest="source", action="store_const", const="url-list")

    parser.add_argument("--n-per-finding", type=int, default=5, help="Stub mode: cases per finding.")
    parser.add_argument("--local-dir", type=Path, help="Local mode: source directory of images.")
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        help="Optional NIH Data_Entry_2017.csv; used by local/url-list modes for labels.",
    )
    parser.add_argument("--url-file", type=Path, help="URL list mode: text file with 'image_index url' per line.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "stub":
        entries = _download_stub(args.raw_dir, args.n_per_finding)
    elif args.source == "local":
        if args.local_dir is None:
            logger.error("--local-dir is required for --source local")
            return 2
        metadata_path = args.metadata_csv or (args.raw_dir / METADATA_FILENAME)
        if not metadata_path.exists():
            logger.error("Metadata CSV not found: %s", metadata_path)
            return 2
        local_metadata = read_metadata_csv(metadata_path)
        entries = _download_local(args.raw_dir, args.local_dir, local_metadata)
    elif args.source == "url-list":
        if args.url_file is None:
            logger.error("--url-file is required for --source url-list")
            return 2
        url_metadata: list[dict[str, str]] | None = None
        if args.metadata_csv is not None:
            url_metadata = read_metadata_csv(args.metadata_csv)
        entries = _download_url_list(args.raw_dir, args.url_file, url_metadata)
    else:
        raise SystemExit(f"Unknown source: {args.source}")

    if not entries:
        logger.error("No files downloaded. Aborting provenance write.")
        return 1

    _write_provenance(entries, args.raw_dir)
    logger.info(
        "Downloaded %d files into %s (source=%s).",
        len(entries),
        args.raw_dir / IMAGES_DIR_NAME,
        args.source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
