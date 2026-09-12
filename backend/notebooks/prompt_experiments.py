"""Prompt experiments for the MedGemma feedback call.

Phase 1 only ships a small set of canned payloads from the dry-run
analyzer. This file is the interactive playground for the next
iteration: vary the system / user prompts against a handful of cases
and compare parsed outputs side-by-side.

It is shipped as a plain Python file (not a `.ipynb`) so it can be
edited, type-checked, and unit-tested with the rest of the codebase.
To run it interactively, paste the cells into a Jupyter notebook or
run them in VS Code's "Run File" interactive window.

The `evaluate_variation(...)` helper does the work; the rest of the
file is example cells.

Usage from the Python REPL or notebook:

    from cxt.infrastructure.data.manifest_repository import ManifestRepository
    from cxt.infrastructure.ml.dry_run_analyzer import DryRunMedGemmaAnalyzer
    from cxt.infrastructure.ml.output_parser import OutputParser
    from cxt.infrastructure.eval import EVAL_SYSTEM_PROMPT, EVAL_USER_PROMPT
    from notebooks.prompt_experiments import evaluate_variation

    cases = ManifestRepository(MANIFEST_PATH).list(split="eval", limit=5)
    analyzer = DryRunMedGemmaAnalyzer()
    parser = OutputParser()
    result = evaluate_variation(
        cases, analyzer, parser,
        system_prompt=EVAL_SYSTEM_PROMPT,
        user_prompt_template=EVAL_USER_PROMPT,
    )
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from cxt.domain.entities.case import Case
from cxt.domain.ports.image_analyzer import IImageAnalyzer
from cxt.domain.value_objects.cxr_image_ref import CxrImageRef
from cxt.infrastructure.eval import EVAL_SYSTEM_PROMPT, EVAL_USER_PROMPT
from cxt.infrastructure.ml.output_parser import OutputParser
from cxt.shared.errors import UpstreamUnavailableError, ValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PromptVariationResult:
    case_id: str
    ground_truth_category: str
    raw_text: str
    parse_succeeded: bool
    predicted_category: str | None
    latency_ms: float
    prompt_tokens: int
    output_tokens: int
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=lambda: {})


@dataclass(frozen=True, slots=True)
class VariationSummary:
    variation_name: str
    system_prompt: str
    user_prompt_template: str
    results: tuple[PromptVariationResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "variation_name": self.variation_name,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "results": [asdict(r) for r in self.results],
        }


def evaluate_variation(
    cases: Sequence[Case],
    analyzer: IImageAnalyzer,
    parser: OutputParser,
    *,
    variation_name: str,
    system_prompt: str,
    user_prompt_template: str,
) -> VariationSummary:
    """Run a single (system, user) prompt pair over a list of cases.

    Returns parsed-vs-truth side-by-side, plus latency and token usage,
    so a human can compare variations at a glance.
    """
    results: list[PromptVariationResult] = []
    for case in cases:
        context = case.metadata.get("clinical_context", "").strip() or "(no clinical context provided)"
        system = system_prompt
        user = user_prompt_template.format(clinical_context=context)
        started = time.perf_counter()
        error: str | None = None
        parse_succeeded = False
        predicted: str | None = None
        prompt_tokens = 0
        output_tokens = 0
        raw_text = ""
        try:
            analysis = analyzer.analyze(
                image=CxrImageRef(uri=case.image_url), prompt=user, system=system
            )
            raw_text = analysis.raw_text
            prompt_tokens = int(analysis.model_output.prompt_tokens)
            output_tokens = int(analysis.model_output.output_tokens)
            try:
                feedback = parser.parse(attempt_id=f"prompt-exp-{case.id}", raw_text=raw_text)
                parse_succeeded = True
                predicted = feedback.likely_finding.value
            except (ValidationError, UpstreamUnavailableError) as exc:
                error = f"parse: {exc.__class__.__name__}: {exc.message}"
        except (UpstreamUnavailableError, ValidationError) as exc:
            error = f"analyze: {exc.__class__.__name__}: {exc.message}"
        except Exception as exc:  # defensive
            error = f"unexpected: {exc.__class__.__name__}: {exc}"
            logger.exception("Unexpected error in prompt experiment for case %s", case.id)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        results.append(
            PromptVariationResult(
                case_id=case.id,
                ground_truth_category=case.ground_truth_finding.category.value,
                raw_text=raw_text,
                parse_succeeded=parse_succeeded,
                predicted_category=predicted,
                latency_ms=elapsed_ms,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                error=error,
            )
        )
    return VariationSummary(
        variation_name=variation_name,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        results=tuple(results),
    )


# ---------------------------------------------------------------------------
# Example variations
# ---------------------------------------------------------------------------
# These are illustrative; the *real* prompt-experiment workflow is to
# copy one of these into a notebook cell, tweak it, and re-run
# `evaluate_variation(...)` against a few cases.
# ---------------------------------------------------------------------------


VARIATION_BASELINE = {
    "name": "baseline (eval prompt)",
    "system": EVAL_SYSTEM_PROMPT,
    "user": EVAL_USER_PROMPT,
}


VARIATION_CONCISE_SYSTEM = (
    "You are a radiology attending. Look at the chest X-ray and respond "
    "with a single JSON object that has `primary_finding.category` and "
    "`primary_finding.confidence` (0-1) only. No prose."
)


VARIATION_VERBOSE_SYSTEM = (
    "You are an expert thoracic radiologist writing feedback for a "
    "medical student. Be thorough: list every key image feature, "
    "compare it to two differential findings, and end with one "
    "teaching point. Output ONLY a single JSON object — no markdown, "
    "no prose, no code fences."
)


VARIATION_TIGHT_USER = (
    "CXR. Clinical context: {clinical_context}. "
    "Primary finding (category + confidence) as JSON."
)


def _baseline_uses_default(cases: Sequence[Case], analyzer: IImageAnalyzer, parser: OutputParser) -> VariationSummary:
    """Special-case the baseline so it uses the same prompts as `run_eval.py`."""
    return evaluate_variation(
        cases,
        analyzer,
        parser,
        variation_name="baseline (matches run_eval.py)",
        system_prompt=EVAL_SYSTEM_PROMPT,
        user_prompt_template=EVAL_USER_PROMPT,
    )


VARIATIONS: dict[str, dict[str, str]] = {
    "baseline": VARIATION_BASELINE,
    "concise": {
        "name": "concise (category + confidence only)",
        "system": VARIATION_CONCISE_SYSTEM,
        "user": VARIATION_TIGHT_USER,
    },
    "verbose": {
        "name": "verbose (full feedback schema)",
        "system": VARIATION_VERBOSE_SYSTEM,
        "user": "{clinical_context}",
    },
}


def render_results_table(summary: VariationSummary) -> str:
    """Return a markdown table summarising the variation's results."""
    lines = [f"## {summary.variation_name}", ""]
    lines.append("| case_id | ground_truth | predicted | parsed | latency_ms | err |")
    lines.append("|---|---|---|---:|---:|---|")
    for r in summary.results:
        err_cell = (r.error or "").replace("|", "\\|")
        predicted = r.predicted_category if r.predicted_category is not None else "-"
        lines.append(
            f"| `{r.case_id}` | `{r.ground_truth_category}` | "
            f"`{predicted}` | "
            f"{'yes' if r.parse_succeeded else 'no'} | "
            f"{r.latency_ms:.0f} | {err_cell} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "evaluate_variation",
    "render_results_table",
    "VARIATIONS",
    "VARIATION_BASELINE",
    "VARIATION_CONCISE_SYSTEM",
    "VARIATION_VERBOSE_SYSTEM",
    "VARIATION_TIGHT_USER",
    "PromptVariationResult",
    "VariationSummary",
]
