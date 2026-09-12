"""Run the Phase 2 feedback evaluation across the five arms.

    python scripts/run_feedback_eval.py --runtime dry_run          # wiring check
    python scripts/run_feedback_eval.py --max-cases 12             # iterate
    python scripts/run_feedback_eval.py --max-cases 30 --out reports/phase2_eval.json

**Run every arm in one invocation, and in one notebook session.** Model and
runtime drift between sessions would confound the single comparison the whole
phase rests on. The script therefore runs all arms in sequence by default
rather than offering a one-arm-at-a-time mode as the happy path.

`p3.0-empty` is always run, even when `--arms` narrows the set, because the
confabulation metric is defined as a difference against it.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cxt.infrastructure.config.settings import (  # noqa: E402
    MedGemmaRuntime,
    Settings,
    get_settings,
)
from cxt.infrastructure.data.manifest_repository import ManifestRepository  # noqa: E402
from cxt.infrastructure.eval.feedback_eval import (  # noqa: E402
    CONTROL_ARM,
    FEEDBACK_ARMS,
    compute_confabulation_rate,
    load_attempt_fixtures,
    render_feedback_summary,
    resolve_feedback_arm,
    run_feedback_evaluation,
)
from cxt.infrastructure.logging.redaction import redact_url  # noqa: E402
from cxt.infrastructure.ml.dry_run_analyzer import DryRunMedGemmaAnalyzer  # noqa: E402
from cxt.infrastructure.ml.image_loader import (  # noqa: E402
    ByReferenceImageLoader,
    LocalImageLoader,
)
from cxt.infrastructure.ml.kaggle_tunnel_client import (  # noqa: E402
    KaggleTunnelMedGemmaClient,
)
from cxt.infrastructure.ml.medgemma_analyzer import MedGemmaAnalyzer  # noqa: E402
from cxt.infrastructure.retrieval.corpus import load_corpus  # noqa: E402

logger = logging.getLogger("run_feedback_eval")

FIXTURES_PATH = REPO_ROOT / "tests" / "fixtures" / "student_attempts.json"


def _build_analyzer(settings: Settings, *, warm: bool):  # type: ignore[no-untyped-def]
    if settings.medgemma_runtime == MedGemmaRuntime.DRY_RUN:
        return DryRunMedGemmaAnalyzer(), ByReferenceImageLoader()
    if not settings.medgemma_tunnel_url:
        raise SystemExit("MEDGEMMA_RUNTIME is kaggle_tunnel but MEDGEMMA_TUNNEL_URL is empty.")
    client = KaggleTunnelMedGemmaClient(settings)
    if not client.ping():
        raise SystemExit(
            f"MedGemma tunnel at {redact_url(settings.medgemma_tunnel_url)} did not "
            "answer /ping. Is the Kaggle notebook awake?"
        )
    if warm and not client.warm_up():
        raise SystemExit("MedGemma warm-up failed; POST /warmup directly to see its detail.")
    return (
        MedGemmaAnalyzer(client=client),
        LocalImageLoader(settings.images_dir, max_bytes=settings.medgemma_max_image_bytes),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 feedback evaluation.")
    parser.add_argument("--max-cases", type=int, default=12, help="samples per arm (default: 12)")
    parser.add_argument("--split", default="test", help="manifest split (default: test)")
    parser.add_argument("--out", default="reports/phase2_eval.json")
    parser.add_argument(
        "--arms",
        nargs="*",
        default=list(FEEDBACK_ARMS),
        choices=list(FEEDBACK_ARMS),
        help="arms to run (default: all five)",
    )
    parser.add_argument(
        "--runtime", choices=("auto", "dry_run", "kaggle_tunnel"), default="auto"
    )
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    logging.basicConfig(
        level=os.environ.get("APP_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    settings = get_settings()
    if args.runtime != "auto":
        runtime = (
            MedGemmaRuntime.DRY_RUN
            if args.runtime == "dry_run"
            else MedGemmaRuntime.KAGGLE_TUNNEL
        )
        settings = settings.model_copy(update={"medgemma_runtime": runtime})

    if not settings.manifest_path.exists():
        raise SystemExit(f"No manifest at {settings.manifest_path}. Run scripts/dev.py real-data.")

    cases = ManifestRepository(settings.manifest_path).list(split=args.split)
    fixtures = load_attempt_fixtures(FIXTURES_PATH)
    corpus = load_corpus(settings.corpus_dir)
    analyzer, image_loader = _build_analyzer(settings, warm=not args.no_warmup)

    # The control is always run: confabulation is defined as a difference
    # against it, so narrowing the arms must not silently drop the baseline.
    arm_names = list(dict.fromkeys([*args.arms, CONTROL_ARM]))
    logger.info(
        "Arms: %s | cases: %d | fixtures: %d | corpus: %s",
        ", ".join(arm_names),
        len(cases),
        len(fixtures),
        corpus.version,
    )

    reports = []
    for name in arm_names:
        arm = resolve_feedback_arm(name)
        logger.info("Running arm %s...", name)
        reports.append(
            run_feedback_evaluation(
                arm=arm,
                cases=cases,
                fixtures=fixtures,
                analyzer=analyzer,
                image_loader=image_loader,
                corpus=corpus,
                max_cases=args.max_cases,
            )
        )

    control = next((r for r in reports if r.arm == CONTROL_ARM), None)
    if control is not None:
        reports = [
            replace(r, confabulation_rate=compute_confabulation_rate(r, control))
            if r.retrieval_mode != "null"
            else r
            for r in reports
        ]

    # Restore the order the user asked for; the control runs regardless but
    # should not jump the queue in the report.
    order = {name: i for i, name in enumerate(arm_names)}
    reports.sort(key=lambda r: order.get(r.arm, 99))

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "corpus_version": corpus.version,
                "split": args.split,
                "max_cases": args.max_cases,
                "control_arm": CONTROL_ARM,
                "arms": [r.to_dict() for r in reports],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = render_feedback_summary(reports)
    out_path.with_suffix(".md").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    print(f"\nWrote {out_path} and {out_path.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
