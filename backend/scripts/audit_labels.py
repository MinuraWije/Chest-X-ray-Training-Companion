"""Audit an eval report against the full NIH multi-label ground truth.

    python scripts/audit_labels.py reports/phase1_eval_e1_bf16_n12.json
    python scripts/audit_labels.py reports/phase1_eval.json --out reports/audit.md

The eval harness scores a prediction against `manifest.finding`, which is a
*single* category chosen by `find_primary_category` from a case that may
carry several NIH labels. 43% of the corpus is multi-label, so a model can
name a finding that is genuinely present and still be scored wrong.

This script recomputes two figures over the same samples:

- **strict**  — prediction == the manifest's primary category (what the
  harness reports).
- **lenient** — prediction is *any* category the NIH labels support.

The gap between them is a property of the label reduction, not of the
model, and both belong in the write-up. Lenient is not the better number:
it cannot distinguish a model that reasoned correctly from one that
guessed a common finding on a case that happens to have four labels. It
bounds how much of the strict error is an artefact of our own mapping.

The script is read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cxt.domain.entities.finding import FindingCategory  # noqa: E402
from cxt.infrastructure.config.settings import get_settings  # noqa: E402
from cxt.infrastructure.data.nih import (  # noqa: E402
    nih_labels_to_finding_categories,
)


@dataclass(frozen=True, slots=True)
class AuditRow:
    """One evaluated case, rescored against every label NIH gives it."""

    case_id: str
    image_index: str
    primary_gt: str
    predicted: str
    all_true: tuple[str, ...]
    nih_labels: str

    @property
    def strict_ok(self) -> bool:
        return self.primary_gt == self.predicted

    @property
    def lenient_ok(self) -> bool:
        return self.predicted in self.all_true

    @property
    def masked(self) -> bool:
        """Scored wrong only because the primary-category reduction hid a true label."""
        return self.lenient_ok and not self.strict_ok


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def build_rows(report: dict[str, object], manifest: dict[str, dict[str, str]]) -> list[AuditRow]:
    samples = report.get("samples")
    if not isinstance(samples, list):
        raise SystemExit("Report has no `samples` array — is this an eval report?")

    rows: list[AuditRow] = []
    for sample in samples:
        case_id = str(sample["case_id"])
        entry = manifest.get(case_id)
        if entry is None:
            raise SystemExit(
                f"Case {case_id} is in the report but not in the manifest. "
                "The manifest was rebuilt after the eval ran; the two no longer "
                "describe the same corpus and the audit would be meaningless."
            )
        predicted = sample.get("predicted_category")
        if predicted is None:
            # A parse failure has no prediction to rescore; skipping it keeps
            # the audit about label quality rather than about parsing.
            continue
        cats = nih_labels_to_finding_categories([entry["nih_labels"]])
        rows.append(
            AuditRow(
                case_id=case_id,
                image_index=entry["image_index"],
                primary_gt=str(sample["ground_truth_category"]),
                predicted=str(predicted),
                all_true=tuple(c.value for c in cats),
                nih_labels=entry["nih_labels"],
            )
        )
    return rows


def render(rows: list[AuditRow], report_path: Path, report: dict[str, object]) -> str:
    total = len(rows)
    if total == 0:
        return "# Label audit\n\nNo parsed predictions to audit.\n"

    strict = sum(r.strict_ok for r in rows)
    lenient = sum(r.lenient_ok for r in rows)
    masked = [r for r in rows if r.masked]
    multi = sum(1 for r in rows if len(r.all_true) > 1)
    pred_counts = Counter(r.predicted for r in rows)

    lines: list[str] = [
        "# Label audit",
        "",
        f"- Source report: `{report_path.name}`",
        f"- Prompt: `{report.get('prompt_version', '?')}`",
        f"- Model: `{report.get('model', '?')}`",
        f"- Cases rescored: {total}",
        "",
        "## Headline",
        "",
        "| Scoring | Correct | Rate |",
        "| --- | --- | --- |",
        f"| Strict (primary category) | {strict}/{total} | {strict / total:.1%} |",
        f"| Lenient (any true NIH label) | {lenient}/{total} | {lenient / total:.1%} |",
        "",
        f"{len(masked)} of the {total - strict} strict errors are cases where the model "
        "named a finding the NIH labels do support; the single-label reduction hid it.",
        "",
        f"{multi}/{total} evaluated cases carry more than one of our six categories.",
        "",
        "## Per case",
        "",
        "| Image | Primary GT | Predicted | Strict | Lenient | All true categories | NIH labels |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row.image_index}` | {row.primary_gt} | {row.predicted} "
            f"| {'yes' if row.strict_ok else 'no'} | {'yes' if row.lenient_ok else 'no'} "
            f"| {', '.join(row.all_true)} | {row.nih_labels} |"
        )

    lines += ["", "## Prediction distribution", "", "| Predicted | Count |", "| --- | --- |"]
    for name, count in pred_counts.most_common():
        lines.append(f"| {name} | {count} |")

    unused = sorted({c.value for c in FindingCategory} - set(pred_counts))
    if unused:
        lines += ["", f"Never predicted: {', '.join(unused)}."]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Path to an eval report .json")
    parser.add_argument("--out", type=Path, default=None, help="Write markdown here")
    parser.add_argument("--manifest", type=Path, default=None, help="Override manifest.csv")
    args = parser.parse_args(argv)

    report_path = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    if not report_path.exists():
        raise SystemExit(f"No such report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    manifest_path = args.manifest or get_settings().manifest_path
    if not manifest_path.exists():
        raise SystemExit(f"No such manifest: {manifest_path}")

    rows = build_rows(report, load_manifest(manifest_path))
    markdown = render(rows, report_path, report)
    print(markdown)

    if args.out is not None:
        out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
