"""Tests for the corrective re-prompt path.

A 4-bit 4B model asked for structured JSON will sometimes answer with prose.
`DEVELOPMENT_PLAN.md` Step 4 asks for one corrective retry before giving up.
"""
from __future__ import annotations

import json

import pytest

from cxt.domain.value_objects.cxr_image_ref import CxrImageRef
from cxt.infrastructure.ml.resilient_analyzer import RetryingFeedbackParser
from cxt.shared.errors import UpstreamUnavailableError
from tests.support import StubAnalyzer, UnavailableAnalyzer
from tests.support.doubles import feedback_payload

_IMAGE = CxrImageRef(uri="/images/c-1.png")
_GOOD = json.dumps(feedback_payload("pneumothorax"))


class _ScriptedAnalyzer:
    """IImageAnalyzer replaying a fixed list of raw responses."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[CxrImageRef, str, str]] = []

    def analyze(self, *, image: CxrImageRef, prompt: str, system: str = ""):  # type: ignore[no-untyped-def]
        from cxt.domain.ports.image_analyzer import ImageAnalysis
        from cxt.domain.ports.medgemma_client import ModelOutput

        self.calls.append((image, prompt, system))
        text = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        return ImageAnalysis(
            raw_text=text,
            model_output=ModelOutput(text=text, prompt_tokens=1, output_tokens=1, model="scripted"),
        )


class TestRetryingFeedbackParser:
    def test_clean_output_does_not_re_prompt(self) -> None:
        analyzer = StubAnalyzer()
        result = RetryingFeedbackParser(analyzer).analyze_and_parse(
            attempt_id="a1", image=_IMAGE, prompt="p", system="s"
        )

        assert result.repair_attempts == 0
        assert len(analyzer.calls) == 1
        assert result.feedback.attempt_id == "a1"

    def test_recovers_on_the_corrective_second_call(self) -> None:
        analyzer = _ScriptedAnalyzer("I think this is a pneumothorax, honestly.", _GOOD)
        result = RetryingFeedbackParser(analyzer).analyze_and_parse(
            attempt_id="a1", image=_IMAGE, prompt="p", system="s"
        )

        assert result.repair_attempts == 1
        assert len(analyzer.calls) == 2
        assert result.feedback.likely_finding.value

    def test_corrective_prompt_keeps_the_original_and_names_the_failure(self) -> None:
        analyzer = _ScriptedAnalyzer("nonsense", _GOOD)
        RetryingFeedbackParser(analyzer).analyze_and_parse(
            attempt_id="a1", image=_IMAGE, prompt="ORIGINAL", system="s"
        )

        _, retry_prompt, retry_system = analyzer.calls[1]
        assert retry_prompt.startswith("ORIGINAL")
        assert "could not be parsed" in retry_prompt
        assert "primary_finding" in retry_prompt
        # The system prompt carries the schema; it must not drift on the retry.
        assert retry_system == "s"

    def test_gives_up_after_the_configured_budget(self) -> None:
        analyzer = _ScriptedAnalyzer("still nonsense")
        with pytest.raises(UpstreamUnavailableError):
            RetryingFeedbackParser(analyzer, max_repair_attempts=2).analyze_and_parse(
                attempt_id="a1", image=_IMAGE, prompt="p", system="s"
            )
        assert len(analyzer.calls) == 3  # first call + 2 repairs

    def test_zero_budget_means_no_retry(self) -> None:
        analyzer = _ScriptedAnalyzer("nonsense", _GOOD)
        with pytest.raises(UpstreamUnavailableError):
            RetryingFeedbackParser(analyzer, max_repair_attempts=0).analyze_and_parse(
                attempt_id="a1", image=_IMAGE, prompt="p", system="s"
            )
        assert len(analyzer.calls) == 1

    def test_a_dead_tunnel_is_not_retried_as_a_parse_failure(self) -> None:
        analyzer = UnavailableAnalyzer()
        with pytest.raises(UpstreamUnavailableError, match="model offline"):
            RetryingFeedbackParser(analyzer).analyze_and_parse(
                attempt_id="a1", image=_IMAGE, prompt="p", system="s"
            )
        # The first analyze() raised, so the retry loop never engaged.
        assert analyzer.calls == 1
