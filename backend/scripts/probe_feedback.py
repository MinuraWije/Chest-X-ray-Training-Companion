"""Run the real student-submission path over a handful of cases and print the feedback.

    python scripts/probe_feedback.py --max-cases 6
    python scripts/probe_feedback.py --max-cases 6 --out reports/probe_p2.md

This is not `run_eval.py`. The eval harness uses its own bias-free prompt
sets (`EVAL_PROMPT_SETS`) to measure whether the model can *recognise* a
finding. That question stopped being on the product's critical path when the
prompts started supplying the verified finding instead of asking for it, so
there is no longer any harness covering what students actually receive.

This script closes that gap: it drives `SubmitAttemptUseCase` — the same use
case the API calls — through `PromptBuilder`, the retrying parser and the
ground-truth guard, and prints each feedback in full so the prose can be read
and judged. Nothing here is scored automatically; the thing that matters about
this output is whether the rationale describes the film in front of it or a
textbook, and no string comparison settles that.

The two numbers it does report:

- **structure validity** — how often the output parsed at all.
- **override rate** — how often the model returned a category other than the
  one it was handed. Non-zero means the prompt is not holding.

Attempts are kept in memory; the script never touches the database.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cxt.application.use_cases.submit_attempt import (  # noqa: E402
    SubmitAttemptRequest,
    SubmitAttemptUseCase,
)
from cxt.domain.entities.feedback import Feedback  # noqa: E402
from cxt.domain.entities.student_attempt import StudentAttempt  # noqa: E402
from cxt.infrastructure.config.settings import MedGemmaRuntime, get_settings  # noqa: E402
from cxt.infrastructure.data.manifest_repository import ManifestRepository  # noqa: E402
from cxt.infrastructure.eval import select_eval_cases  # noqa: E402
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
from cxt.infrastructure.ml.prompts import (  # noqa: E402
    PromptBuilder,
    load_default_prompts,
)
from cxt.infrastructure.retrieval.null_retriever import NullRetriever  # noqa: E402
from cxt.shared.errors import UpstreamUnavailableError  # noqa: E402

logger = logging.getLogger("probe_feedback")

# A deliberately mediocre answer: plausible, non-empty, and wrong often enough
# that the "comparison with student answer" section has something to do. Using
# the same text for every case keeps the feedback differences attributable to
# the image rather than to the student input.
DEFAULT_STUDENT_TEXT = (
    "The lungs look mostly clear to me. I think the heart size is normal and "
    "I cannot see any obvious collapse or fluid at the bases."
)


class InMemoryAttemptRepo:
    """IAttemptRepository that forgets everything. The probe is read-only."""

    def __init__(self) -> None:
        self._attempts: dict[str, StudentAttempt] = {}
        self._feedback: dict[str, Feedback] = {}

    def add(self, attempt: StudentAttempt) -> None:
        self._attempts[attempt.id] = attempt

    def get(self, attempt_id: str) -> StudentAttempt:
        return self._attempts[attempt_id]

    def list_for_student(self, student_id: str) -> list[StudentAttempt]:
        return [a for a in self._attempts.values() if a.student_id == student_id]

    def list_for_educator_review(self) -> list[StudentAttempt]:
        return list(self._attempts.values())

    def attach_feedback(self, attempt_id: str, feedback: Feedback) -> None:
        self._feedback[attempt_id] = feedback

    def get_with_feedback(self, attempt_id: str) -> tuple[StudentAttempt, Feedback | None]:
        return self._attempts[attempt_id], self._feedback.get(attempt_id)


def _render(case_no: int, case_id: str, truth: str, feedback: Feedback, ms: int) -> str:
    raw = feedback.raw
    overridden = bool(raw.get("ground_truth_override"))
    lines = [
        f"### {case_no}. `{case_id}` — confirmed: **{truth}**",
        "",
        f"- Latency: {ms} ms",
        f"- Model confidence in the demonstration: {feedback.confidence.value:.2f}",
    ]
    if overridden:
        lines.append(
            f"- **OVERRIDE**: model answered `{raw.get('model_category')}` despite being "
            f"told `{truth}`."
        )
    lines += [
        "",
        f"**Displayed finding:** {feedback.likely_finding.value}",
        "",
        f"**Key signs (rationale):** {feedback.key_signs.value}",
        "",
        f"**Key observations:** "
        f"{feedback.key_observations.value if feedback.key_observations else '(none)'}",
        "",
        f"**Comparison with the student:** {feedback.comparison.value}",
        "",
        f"**Missed clues:** {feedback.missed_clues.value}",
        "",
        f"**Teaching point:** {feedback.learning_point.value}",
        "",
    ]
    return "\n".join(lines)


def _build_analyzer(settings, *, warm: bool):  # type: ignore[no-untyped-def]
    if settings.medgemma_runtime == MedGemmaRuntime.DRY_RUN:
        return DryRunMedGemmaAnalyzer(), ByReferenceImageLoader()
    if not settings.medgemma_tunnel_url:
        raise SystemExit("MEDGEMMA_RUNTIME is kaggle_tunnel but MEDGEMMA_TUNNEL_URL is empty.")
    client = KaggleTunnelMedGemmaClient(settings)
    if not client.ping():
        raise SystemExit(
            f"MedGemma tunnel at {redact_url(settings.medgemma_tunnel_url)} did not answer /ping. "
            "Is the Kaggle notebook awake? Restarting it changes the ngrok URL."
        )
    if warm and not client.warm_up():
        raise SystemExit(
            "MedGemma warm-up failed. The notebook is reachable but the model is "
            "not answering — POST /warmup directly to see its `detail`."
        )
    return (
        MedGemmaAnalyzer(client=client),
        LocalImageLoader(settings.images_dir, max_bytes=settings.medgemma_max_image_bytes),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cases", type=int, default=6)
    parser.add_argument("--split", default="test")
    parser.add_argument("--student-text", default=DEFAULT_STUDENT_TEXT)
    parser.add_argument("--out", type=Path, default=None, help="also write markdown here")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    analyzer, image_loader = _build_analyzer(settings, warm=not args.no_warmup)

    cases = select_eval_cases(
        ManifestRepository(settings.manifest_path),
        split=args.split,
        max_cases=args.max_cases,
    )
    if not cases:
        raise SystemExit(f"No cases in split {args.split!r}. Has the manifest been built?")

    # Built here rather than left to the use case's default purely so the
    # report can name the prompt version without reaching into private state.
    prompt_builder = PromptBuilder(templates=load_default_prompts())
    use_case = SubmitAttemptUseCase(
        analyzer=analyzer,
        attempt_repository=InMemoryAttemptRepo(),
        case_repository=ManifestRepository(settings.manifest_path),
        retriever=NullRetriever(),
        image_loader=image_loader,
        prompt_builder=prompt_builder,
    )

    blocks: list[str] = []
    failures = 0
    overrides = 0
    for index, case in enumerate(cases, start=1):
        truth = case.ground_truth_finding.category.value
        logger.info("[%d/%d] %s (%s)", index, len(cases), case.id, truth)
        started = time.perf_counter()
        try:
            result = use_case.execute(
                SubmitAttemptRequest(
                    case_id=case.id, student_id="probe", free_text=args.student_text
                )
            )
        except UpstreamUnavailableError as exc:
            failures += 1
            logger.warning("  failed: %s", exc)
            blocks.append(f"### {index}. `{case.id}` — confirmed: **{truth}**\n\nFAILED: {exc}\n")
            continue
        elapsed = int((time.perf_counter() - started) * 1000)
        if result.feedback.raw.get("ground_truth_override"):
            overrides += 1
        blocks.append(_render(index, case.id, truth, result.feedback, elapsed))

    total = len(cases)
    parsed = total - failures
    summary = "\n".join(
        [
            "# Feedback probe (production path)",
            "",
            f"- Prompt: `{prompt_builder.version}`",
            f"- Split: `{args.split}`  |  Cases: {total}",
            f"- Structure validity: {parsed}/{total} ({parsed / total:.1%})",
            f"- Ground-truth overrides: {overrides}/{parsed}"
            + (" — the prompt is not holding" if overrides else ""),
            "",
            "Read the prose below for grounding: does each rationale describe *this*",
            "film — side, zone, the actual sign — or could it have been written from",
            "the label alone? That is the question this probe exists to answer and",
            "the one no number here settles.",
            "",
        ]
    )
    document = summary + "\n" + "\n".join(blocks)
    print(document)
    if args.out is not None:
        out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(document, encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
