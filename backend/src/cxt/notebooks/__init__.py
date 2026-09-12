"""Re-export of the prompt-experiments helper for `cxt.notebooks.*` imports.

The actual implementation lives at `backend/notebooks/prompt_experiments.py`
so it can live next to the data, settings, and scripts that it
interactively drives. We re-export it from `cxt.notebooks` to keep
imports consistent with the rest of the package layout.
"""
from __future__ import annotations

import importlib.util as _ilu
import sys as _sys
from pathlib import Path
from types import ModuleType

_NOTEBOOKS_DIR = Path(__file__).resolve().parents[3] / "notebooks"


def _load() -> ModuleType:
    spec = _ilu.spec_from_file_location("cxt._notebooks_impl", _NOTEBOOKS_DIR / "prompt_experiments.py")
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError("Could not locate notebooks/prompt_experiments.py")
    module = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_impl: ModuleType = _load()

evaluate_variation = _impl.evaluate_variation
render_results_table = _impl.render_results_table
PromptVariationResult = _impl.PromptVariationResult
VariationSummary = _impl.VariationSummary
VARIATIONS = _impl.VARIATIONS
VARIATION_BASELINE = _impl.VARIATION_BASELINE
VARIATION_CONCISE_SYSTEM = _impl.VARIATION_CONCISE_SYSTEM
VARIATION_VERBOSE_SYSTEM = _impl.VARIATION_VERBOSE_SYSTEM
VARIATION_TIGHT_USER = _impl.VARIATION_TIGHT_USER


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
