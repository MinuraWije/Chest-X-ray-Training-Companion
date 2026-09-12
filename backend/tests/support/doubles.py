"""Hand-rolled protocol-shaped test doubles.

The repo deliberately avoids `unittest.mock`: a double that has to satisfy a
`Protocol` catches signature drift at type-check time, where a `MagicMock`
silently absorbs it.
"""
from __future__ import annotations

import json
import time

from cxt.domain.entities.case import Case
from cxt.domain.entities.finding import Finding, FindingCategory
from cxt.domain.ports.image_analyzer import ImageAnalysis
from cxt.domain.ports.medgemma_client import ModelOutput
from cxt.domain.value_objects.cxr_image_ref import CxrImageRef
from cxt.domain.value_objects.finding_label import FindingLabel
from cxt.shared.errors import NotFoundError, UpstreamUnavailableError


def make_case(
    case_id: str,
    category: FindingCategory,
    *,
    split: str = "test",
    clinical_context: str = "demo context",
    nih_labels: str | None = None,
    image_index: str | None = None,
) -> Case:
    metadata: dict[str, object] = {"clinical_context": clinical_context}
    if image_index is not None:
        # NIH's `<patient>_<follow-up>` naming, which is where `Case.patient_id`
        # comes from. Omitted by default so a test that does not care exercises
        # the unknown-patient path.
        metadata["image_index"] = image_index
    if nih_labels is not None:
        # Pipe-delimited, exactly as the manifest stores it. Only set it when a
        # test cares: a case without it exercises the fallback where lenient
        # scoring degrades to strict.
        metadata["nih_labels"] = nih_labels
    return Case(
        id=case_id,
        image_url=f"/images/{case_id}.png",
        ground_truth_finding=Finding(
            category=category,
            label=FindingLabel(category.value.replace("_", " ").title()),
        ),
        split=split,
        metadata=metadata,
    )


def feedback_payload(category: str = "pneumothorax") -> dict[str, object]:
    """A well-formed model response for `category`."""
    return {
        "primary_finding": {
            "category": category,
            "label": category.replace("_", " ").title(),
            "confidence": 0.8,
            "rationale": "Stub rationale describing the salient image features.",
        },
        "differential_findings": [{"label": "Alternative", "confidence": 0.1}],
        "key_observations": "Stub observations about what is visible on the film.",
        "teaching_point": "Stub teaching point for the student.",
    }


class StubCaseRepo:
    """ICaseRepository over an in-memory list."""

    def __init__(self, cases: list[Case]) -> None:
        self._cases = list(cases)

    def list(
        self,
        *,
        split: str | None = None,
        finding: FindingCategory | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Case]:
        items = [c for c in self._cases if split is None or c.split == split]
        if finding is not None:
            items = [c for c in items if c.ground_truth_finding.category == finding]
        items = items[offset:]
        return items if limit is None else items[:limit]

    def count(self, *, split: str | None = None, finding: FindingCategory | None = None) -> int:
        return len(self.list(split=split, finding=finding))

    def get(self, case_id: str) -> Case:
        for case in self._cases:
            if case.id == case_id:
                return case
        raise NotFoundError(f"Case not found: {case_id}")


class StubAnalyzer:
    """IImageAnalyzer returning a category chosen per image URI.

    `mapping` keys are image URIs; anything unmapped falls back to
    `default_category`. Records every call so tests can assert on the prompts.
    """

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        *,
        default_category: str = "pneumothorax",
        raw_text: str | None = None,
        sleep_s: float = 0.001,
        model: str = "stub",
        models: list[str] | None = None,
    ) -> None:
        self._mapping = mapping or {}
        self._default = default_category
        self._raw_text = raw_text
        self._sleep_s = sleep_s
        self._model = model
        # A per-call sequence, for the case where a run spans more than one
        # model (e.g. the notebook was restarted mid-run on a different image).
        self._models = list(models) if models else None
        self.calls: list[tuple[CxrImageRef, str, str]] = []

    def analyze(self, *, image: CxrImageRef, prompt: str, system: str = "") -> ImageAnalysis:
        self.calls.append((image, prompt, system))
        # Non-zero so latency percentiles are meaningful.
        time.sleep(self._sleep_s)
        if self._raw_text is not None:
            text = self._raw_text
        else:
            category = self._mapping.get(image.uri, self._default)
            text = json.dumps(feedback_payload(category))
        if self._models:
            model = self._models[(len(self.calls) - 1) % len(self._models)]
        else:
            model = self._model
        return ImageAnalysis(
            raw_text=text,
            model_output=ModelOutput(
                text=text,
                prompt_tokens=len(prompt.split()) + len(system.split()),
                output_tokens=len(text.split()),
                model=model,
            ),
        )


class UnavailableAnalyzer:
    """IImageAnalyzer that always fails, for the error paths."""

    def __init__(self, message: str = "model offline") -> None:
        self._message = message
        self.calls = 0

    def analyze(self, *, image: CxrImageRef, prompt: str, system: str = "") -> ImageAnalysis:
        self.calls += 1
        raise UpstreamUnavailableError(self._message)


class FakeMedGemmaClient:
    """IMedGemmaClient double — the port level `MedGemmaAnalyzer` depends on.

    `responses` is consumed one entry per call; the last entry repeats once
    exhausted. An entry that is an exception is raised instead of returned.
    """

    def __init__(self, *responses: str | Exception, model: str = "fake-medgemma") -> None:
        self._responses: list[str | Exception] = list(responses) or [
            json.dumps(feedback_payload())
        ]
        self._model = model
        self.calls: list[tuple[CxrImageRef, str, str]] = []

    def generate(self, *, image: CxrImageRef, prompt: str, system: str = "") -> ModelOutput:
        self.calls.append((image, prompt, system))
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        response = self._responses[index]
        if isinstance(response, Exception):
            raise response
        return ModelOutput(
            text=response,
            prompt_tokens=len(prompt.split()) + len(system.split()),
            output_tokens=len(response.split()),
            model=self._model,
        )
