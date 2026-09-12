"""RetryingFeedbackParser — analyze, parse, and re-prompt once if the JSON is unusable.

`DEVELOPMENT_PLAN.md` Step 4 calls for "one retry with a corrective prompt".
A 4-bit 4B model asked for a four-field JSON object will occasionally answer
with prose, or stop mid-object at the token limit. `OutputParser` repairs what
it can statically; when that fails the cheapest remaining fix is to show the
model its own bad output and ask again.

The retry lives here rather than inside `OutputParser` so the parser stays a
pure text -> entity function and the golden tests keep exercising it directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from cxt.domain.entities.feedback import Feedback
from cxt.domain.ports.image_analyzer import IImageAnalyzer
from cxt.domain.ports.medgemma_client import ModelOutput
from cxt.domain.value_objects.cxr_image_ref import CxrImageRef
from cxt.infrastructure.ml.output_parser import OutputParser
from cxt.shared.errors import UpstreamUnavailableError

logger = logging.getLogger(__name__)

_CORRECTION_TEMPLATE = """Your previous reply could not be parsed as JSON.

The error was: {error}

Reply again with ONLY the JSON object — no prose, no code fences, no commentary
before or after it. It must contain exactly these keys: primary_finding (with
category, label, confidence, rationale), differential_findings, key_observations,
teaching_point. Keep every string short so the object is complete."""


@dataclass(frozen=True, slots=True)
class ParsedAnalysis:
    feedback: Feedback
    model_output: ModelOutput
    repair_attempts: int


class RetryingFeedbackParser:
    def __init__(
        self,
        analyzer: IImageAnalyzer,
        parser: OutputParser | None = None,
        *,
        max_repair_attempts: int = 1,
    ) -> None:
        self._analyzer = analyzer
        self._parser = parser or OutputParser()
        self._max_repair_attempts = max(0, max_repair_attempts)

    def analyze_and_parse(
        self,
        *,
        attempt_id: str,
        image: CxrImageRef,
        prompt: str,
        system: str,
    ) -> ParsedAnalysis:
        analysis = self._analyzer.analyze(image=image, prompt=prompt, system=system)
        try:
            feedback = self._parser.parse(attempt_id=attempt_id, raw_text=analysis.raw_text)
            return ParsedAnalysis(feedback, analysis.model_output, repair_attempts=0)
        except UpstreamUnavailableError as first_error:
            last_error = first_error

        for attempt in range(1, self._max_repair_attempts + 1):
            logger.info(
                "Re-prompting MedGemma for attempt %s (repair %d/%d): %s",
                attempt_id,
                attempt,
                self._max_repair_attempts,
                last_error,
            )
            corrective = f"{prompt}\n\n{_CORRECTION_TEMPLATE.format(error=last_error)}"
            try:
                analysis = self._analyzer.analyze(
                    image=image, prompt=corrective, system=system
                )
                feedback = self._parser.parse(
                    attempt_id=attempt_id, raw_text=analysis.raw_text
                )
            except UpstreamUnavailableError as exc:
                last_error = exc
                continue
            return ParsedAnalysis(feedback, analysis.model_output, repair_attempts=attempt)

        raise last_error
