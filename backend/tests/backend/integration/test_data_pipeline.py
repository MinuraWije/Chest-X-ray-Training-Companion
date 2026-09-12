from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

BACKEND_ROOT = Path(__file__).resolve().parents[3]
PY = BACKEND_ROOT / ".venv" / "Scripts" / "python.exe"


def _python() -> str:
    if PY.exists():
        return str(PY)
    return sys.executable


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_python(), "-m", *args],
        cwd=str(BACKEND_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_stub_download_writes_provenance(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    result = _run(
        [
            "scripts.download_nih_subset",
            "--source",
            "stub",
            "--raw-dir",
            str(raw_dir),
            "--n-per-finding",
            "2",
        ]
    )
    assert result.returncode == 0, result.stderr

    provenance = json.loads((raw_dir / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["dataset"] == "NIH ChestX-ray8 / NIHCC"
    files = provenance["files"]
    assert len(files) == 14

    findings = {f["primary_category"] for f in files}
    assert "no_finding" in findings
    assert "pneumothorax" in findings
    assert "fracture" in findings

    images_dir = raw_dir / "images"
    assert images_dir.is_dir()
    assert any(images_dir.glob("*.png"))


def test_build_manifest_produces_balanced_csv(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    dl = _run(
        [
            "scripts.download_nih_subset",
            "--source",
            "stub",
            "--raw-dir",
            str(raw_dir),
            "--n-per-finding",
            "3",
        ]
    )
    assert dl.returncode == 0, dl.stderr

    build = _run(
        [
            "scripts.build_manifest",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
            "--n-per-finding",
            "3",
            "--n-negatives",
            "3",
        ]
    )
    assert build.returncode == 0, build.stderr

    manifest_path = processed_dir / "manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 21

    by_finding: dict[str, int] = {}
    for row in rows:
        by_finding[row["finding"]] = by_finding.get(row["finding"], 0) + 1
    for finding in (
        "pneumothorax",
        "pleural_effusion",
        "cardiomegaly",
        "consolidation",
        "nodule_or_mass",
        "fracture",
    ):
        assert by_finding.get(finding) == 3, finding
    assert by_finding.get("no_finding") == 3

    splits = {row["split"] for row in rows}
    assert splits.issubset({"train", "val", "test"})

    images_dir = processed_dir / "images"
    assert any(images_dir.glob("*.png"))

    index = json.loads((processed_dir / "case_index.json").read_text(encoding="utf-8"))
    assert index["case_count"] == 21


def test_local_source_reads_nih_layout(tmp_path: Path) -> None:
    """End-to-end: fake a small NIH-style local copy (image_index, Data_Entry CSV)
    and confirm download_nih_subset + build_manifest produce a real-data manifest."""
    local_dir = tmp_path / "local_nih"
    local_dir.mkdir()

    rows = [
        ("00000001_000.png", "Pneumothorax"),
        ("00000002_000.png", "Effusion"),
        ("00000003_000.png", "Cardiomegaly"),
        ("00000004_000.png", "Consolidation"),
        ("00000005_000.png", "Nodule"),
        ("00000006_000.png", "Fracture"),
        ("00000007_000.png", "No Finding"),
        ("00000008_000.png", "Pneumothorax"),
        ("00000009_000.png", "Effusion"),
        ("00000010_000.png", "Infiltration"),
        ("00000011_000.png", "Pneumonia"),
        ("00000012_000.png", "Mass"),
    ]
    for filename, _ in rows:
        Image.new("RGB", (1024, 1024), (40, 40, 50)).save(local_dir / filename)

    metadata_csv = tmp_path / "Data_Entry_2017.csv"
    with metadata_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
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
            ]
        )
        for filename, label in rows:
            writer.writerow(
                [filename, label, "0", "1", "55", "M", "PA", "1024", "1024", "0.143", "0.143"]
            )

    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    dl = _run(
        [
            "scripts.download_nih_subset",
            "--source",
            "local",
            "--raw-dir",
            str(raw_dir),
            "--local-dir",
            str(local_dir),
            "--metadata-csv",
            str(metadata_csv),
            "--n-per-finding",
            "2",
        ]
    )
    assert dl.returncode == 0, dl.stderr

    provenance = json.loads((raw_dir / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["dataset"] == "NIH ChestX-ray8 / NIHCC"
    assert len(provenance["files"]) == len(rows)
    for entry in provenance["files"]:
        assert entry["source"].startswith("local:")

    build = _run(
        [
            "scripts.build_manifest",
            "--raw-dir",
            str(raw_dir),
            "--processed-dir",
            str(processed_dir),
            "--n-per-finding",
            "2",
            "--n-negatives",
            "2",
        ]
    )
    assert build.returncode == 0, build.stderr

    with (processed_dir / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        manifest_rows = list(csv.DictReader(fh))
    assert len(manifest_rows) == 11

    by_finding: dict[str, int] = {}
    for r in manifest_rows:
        by_finding[r["finding"]] = by_finding.get(r["finding"], 0) + 1
    assert by_finding.get("pneumothorax") == 2
    assert by_finding.get("pleural_effusion") == 2
    assert by_finding.get("cardiomegaly") == 1
    assert by_finding.get("consolidation") == 2
    assert by_finding.get("nodule_or_mass") == 2
    assert by_finding.get("fracture") == 1
    assert by_finding.get("no_finding") == 1

    label_sources = {r["label_source"] for r in manifest_rows}
    assert all(src.startswith("local:") for src in label_sources), label_sources

    nih_labels_concat = "|".join(r["nih_labels"] for r in manifest_rows)
    for needle in ("Pneumothorax", "Effusion", "Cardiomegaly", "No Finding"):
        assert needle in nih_labels_concat, needle
