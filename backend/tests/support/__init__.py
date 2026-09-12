"""Shared test doubles.

These were previously duplicated across `test_eval_harness.py`,
`test_prompt_experiments.py` and `test_submit_attempt.py`. They live here so a
port signature change is a one-file edit rather than a scavenger hunt.
"""
from __future__ import annotations

from tests.support.doubles import (
    FakeMedGemmaClient,
    StubAnalyzer,
    StubCaseRepo,
    UnavailableAnalyzer,
    make_case,
)

__all__ = [
    "FakeMedGemmaClient",
    "StubAnalyzer",
    "StubCaseRepo",
    "UnavailableAnalyzer",
    "make_case",
]
