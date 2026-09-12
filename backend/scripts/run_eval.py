"""Run the Phase 1 evaluation harness.

    python scripts/run_eval.py
    python scripts/run_eval.py --max-cases 10 --split eval
    python scripts/run_eval.py --out reports/phase1_eval.json
    python scripts/run_eval.py --runtime kaggle_tunnel   # only if MEDGEMMA_TUNNEL_URL is set
    python scripts/run_eval.py --prompt-version e2.0     # A/B a different eval prompt

The harness calls `IImageAnalyzer.analyze(...)` on each case in the
`eval` split (or `--split`), caps at `--max-cases` (default =
`Settings.max_eval_cases = 30`), and writes:

- `<out>.json`  — full machine-readable report
- `<out>.md`    — human-readable summary

Cases are sampled evenly across ground-truth findings by default, so the
per-finding table stays readable; `--no-stratify` restores the old
first-N-manifest-rows behaviour. Both the prompt version and the sampling
strategy are recorded in the report, so two runs can be compared honestly.

The script is read-only: it does not write to the DB.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
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
from cxt.infrastructure.data.database import (  # noqa: E402
    build_engine,
    init_database,
)
from cxt.infrastructure.data.manifest_repository import (  # noqa: E402
    ManifestRepository,
)
from cxt.infrastructure.eval import (  # noqa: E402
    DEFAULT_EVAL_PROMPT_VERSION,
    EVAL_PROMPT_SETS,
    resolve_prompt_set,
    run_evaluation,
    write_json_report,
    write_markdown_summary,
)
from cxt.infrastructure.logging.redaction import redact_url  # noqa: E402
from cxt.infrastructure.ml.dry_run_analyzer import (  # noqa: E402
    DryRunMedGemmaAnalyzer,
)
from cxt.infrastructure.ml.image_loader import (  # noqa: E402
    ByReferenceImageLoader,
    LocalImageLoader,
)
from cxt.infrastructure.ml.kaggle_tunnel_client import (  # noqa: E402
    KaggleTunnelMedGemmaClient,
)
from cxt.infrastructure.ml.medgemma_analyzer import (  # noqa: E402
    MedGemmaAnalyzer,
)
from cxt.infrastructure.ml.output_parser import (  # noqa: E402
    OutputParser,
)
from cxt.shared.errors import UpstreamUnavailableError  # noqa: E402

logger = logging.getLogger("run_eval")


def _build_analyzer(settings: Settings, *, warm: bool = True):  # type: ignore[no-untyped-def]
    """Return (analyzer, runtime_label, image_loader) for the configured runtime."""
    if settings.medgemma_runtime == MedGemmaRuntime.DRY_RUN:
        return DryRunMedGemmaAnalyzer(), "dry_run", ByReferenceImageLoader()
    if not settings.medgemma_tunnel_url:
        raise SystemExit(
            "MEDGEMMA_RUNTIME is kaggle_tunnel but MEDGEMMA_TUNNEL_URL is empty. "
            "Either paste the ngrok URL into backend/.env or set MEDGEMMA_RUNTIME=dry_run."
        )
    client = KaggleTunnelMedGemmaClient(settings)
    if not client.ping():
        raise SystemExit(
            f"MedGemma tunnel at {redact_url(settings.medgemma_tunnel_url)} did not answer /ping. "
            "Is the Kaggle notebook awake? Restarting it changes the ngrok URL."
        )
    if warm:
        # Loading a 4-bit checkpoint takes minutes; pay it once here rather
        # than letting the first case time out and skew the latency figures.
        logger.info("Warming the model (this can take several minutes on a cold notebook)...")
        if not client.warm_up():
            # Not just slowness: a model loaded at a dtype it cannot generate in
            # returns 200 with empty text for every case, so a full run can
            # complete and report 0% having burned the GPU quota for nothing.
            raise SystemExit(
                "MedGemma warm-up failed. The notebook is reachable but the model "
                "is not answering — check the notebook log, and POST /warmup "
                "directly to see its `detail`. Pass --no-warmup to run anyway."
            )
    return (
        MedGemmaAnalyzer(client=client),
        "kaggle_tunnel",
        LocalImageLoader(settings.images_dir, max_bytes=settings.medgemma_max_image_bytes),
    )


def main(argv: list[str] | None = None) -> int:
    arg_parser = argparse.ArgumentParser(description="Run the Phase 1 evaluation harness.")
    arg_parser.add_argument("--max-cases", type=int, default=None, help="cap on cases (default: settings.max_eval_cases)")
    arg_parser.add_argument("--split", default="test", help="manifest split to evaluate (default: test)")
    arg_parser.add_argument("--out", default="reports/phase1_eval.json", help="output path (default: reports/phase1_eval.json)")
    arg_parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="skip the human-readable .md summary alongside the JSON",
    )
    arg_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the /warmup call before the first case (kaggle_tunnel only)",
    )
    arg_parser.add_argument(
        "--runtime",
        choices=("auto", "dry_run", "kaggle_tunnel"),
        default="auto",
        help="override the analyzer runtime (default: use settings.medgemma_runtime)",
    )
    arg_parser.add_argument(
        "--prompt-version",
        choices=tuple(sorted(EVAL_PROMPT_SETS)),
        default=DEFAULT_EVAL_PROMPT_VERSION,
        help=f"eval prompt set to use (default: {DEFAULT_EVAL_PROMPT_VERSION})",
    )
    arg_parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="take the first --max-cases manifest rows instead of sampling evenly "
        "across ground-truth findings (reproduces pre-stratification runs)",
    )
    args = arg_parser.parse_args(argv if argv is not None else sys.argv[1:])

    logging.basicConfig(
        level=os.environ.get("APP_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    settings = get_settings()
    if args.max_cases is not None:
        settings = settings.model_copy(update={"max_eval_cases": args.max_cases})
    if args.runtime != "auto":
        runtime = MedGemmaRuntime.DRY_RUN if args.runtime == "dry_run" else MedGemmaRuntime.KAGGLE_TUNNEL
        settings = settings.model_copy(update={"medgemma_runtime": runtime})

    if not settings.manifest_path.exists():
        raise SystemExit(
            f"No manifest at {settings.manifest_path}. "
            "Run `python scripts/dev.py data` first to build the eval split."
        )

    if settings.database_url.startswith("sqlite"):
        engine = build_engine(settings)
        init_database(engine)

    case_repo = ManifestRepository(settings.manifest_path)
    analyzer, runtime_label, image_loader = _build_analyzer(settings, warm=not args.no_warmup)
    output_parser = OutputParser()

    try:
        report = run_evaluation(
            case_repository=case_repo,
            analyzer=analyzer,
            parser=output_parser,
            max_cases=settings.max_eval_cases,
            runtime_label=runtime_label,
            split=args.split,
            image_loader=image_loader,
            prompts=resolve_prompt_set(args.prompt_version),
            stratified=not args.no_stratify,
        )
    except UpstreamUnavailableError as exc:
        raise SystemExit(f"Eval failed: {exc}") from exc

    out_path = Path(args.out).resolve()
    write_json_report(report, out_path)
    print(f"Wrote {out_path}")

    if not args.no_markdown:
        md_path = out_path.with_suffix(".md")
        write_markdown_summary(report, md_path)
        print(f"Wrote {md_path}")

    print()
    print(
        f"cases={report.total_cases}  parsed={report.parse_succeeded}  "
        f"prompt={report.prompt_version}  sampling={report.sampling}  "
        f"validity={report.structure_validity * 100:.1f}%  "
        f"recognition={report.finding_recognition * 100:.1f}%  "
        f"latency_p95={report.latency_p95_ms:.0f}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
