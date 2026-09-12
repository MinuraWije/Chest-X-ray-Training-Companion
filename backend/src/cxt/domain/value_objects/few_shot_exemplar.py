"""A labelled image shown to the model before the case under test.

Few-shot prompting on a vision model means putting real radiographs in the
conversation, not just describing them: text-only exemplars can teach the
output *format* (already at 100% structure validity) but nothing about visual
discrimination, which is where the model actually fails.

`answer` is the response the model should have produced for `image`. It is
authored, not model-generated — these are teaching exemplars, and a wrong one
would teach the wrong thing.
"""
from __future__ import annotations

from dataclasses import dataclass

from cxt.domain.value_objects.cxr_image_ref import CxrImageRef
from cxt.shared.errors import ValidationError


@dataclass(frozen=True, slots=True)
class FewShotExemplar:
    image: CxrImageRef
    answer: str

    def __post_init__(self) -> None:
        if not isinstance(self.image, CxrImageRef):
            raise ValidationError(
                f"FewShotExemplar.image must be a CxrImageRef, got {type(self.image).__name__}."
            )
        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ValidationError("FewShotExemplar.answer must be a non-empty string.")
