"""Phase 1 evaluation harness.

This is intentionally narrow (see `docs/EVALUATION_RUBRIC.md`):

- **Structure validity** — % of outputs that parsed to a valid `Feedback`.
- **Finding recognition** — % of cases where `feedback.likely_finding`
  matches the case's `ground_truth_finding.category`.
- **Per-finding breakdown** — finding recognition split by ground-truth
  category (NIH labels are noisy, so a flat accuracy is misleading).
- **Latency** — per-case wall-clock around the analyzer call.
  Reports p50, p95, and mean.
- **Token usage** — input + output tokens per case, summed across cases.
  Available for both the dry-run analyzer (estimates) and the
  Kaggle-tunnel analyzer (real counts from the model).

The harness depends only on the existing ports:
`ICaseRepository`, `IImageAnalyzer`, and the `OutputParser`. It does
**not** write to the DB. It does **not** call the `SubmitAttemptUseCase`
(it would persist and validate a free-text field — neither is what we
want to measure here). It calls the analyzer directly with a fixed
"describe what you see" prompt so the metrics reflect the **model**, not
the student's writing.

Phase 2 will add: retrieval relevance, evidence grounding, and
learning-usefulness metrics. See `docs/EVALUATION_RUBRIC.md`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from cxt.domain.entities.case import Case
from cxt.domain.ports.case_repository import ICaseRepository
from cxt.domain.ports.image_analyzer import IImageAnalyzer
from cxt.domain.ports.image_loader import IImageLoader
from cxt.infrastructure.data.nih import nih_labels_to_finding_categories
from cxt.infrastructure.ml.image_loader import ByReferenceImageLoader
from cxt.infrastructure.ml.output_parser import OutputParser
from cxt.shared.errors import UpstreamUnavailableError, ValidationError

logger = logging.getLogger(__name__)


# Fixed evaluation prompt. Deliberately bias-free: the model gets
# clinical context only and is asked to describe what it sees, not to
# agree with a (non-existent) student. The per-case `clinical_context`
# is the only variable input.
EVAL_SYSTEM_PROMPT = (
    "You are an expert thoracic radiologist reviewing a chest X-ray "
    "as part of an internal evaluation. Output ONLY a single JSON object "
    "(no markdown, no prose, no code fences) with the following keys:\n\n"
    "{\n"
    '  "primary_finding": {\n'
    '    "category": "<one of: pneumothorax, pleural_effusion, '
    'cardiomegaly, consolidation, nodule_or_mass, fracture, no_finding>",\n'
    '    "label": "<short human-readable label>",\n'
    '    "confidence": <number between 0.0 and 1.0>,\n'
    '    "rationale": "<1-3 sentence rationale citing the image features '
    'that support the diagnosis>"\n'
    "  },\n"
    '  "differential_findings": [\n'
    '    {"label": "<differential 1 label>", "confidence": <0.0-1.0>}\n'
    "  ],\n"
    '  "key_observations": "<2-4 sentences describing the specific '
    'features on the image that drove the diagnosis>",\n'
    '  "teaching_point": "<1-2 sentence educational takeaway>"\n'
    "}\n\n"
    "Be specific to what you see on the image; do not invent features."
)

EVAL_USER_PROMPT = (
    "Examine the chest radiograph. Clinical context: {clinical_context}. "
    "Respond with the JSON object described above."
)

# --- e2.0 -------------------------------------------------------------------
# The e1.0 run called `pneumothorax` on 15 of 30 cases when only 6 were: the
# model was not guessing outside the label set, it was over-committing inside
# it. Two changes target that. First, each category is given the sign that must
# actually be visible before it may be called, so "lucent-looking" is no longer
# sufficient evidence for a pneumothorax. Second, `key_observations` is asked
# for FIRST, so the model describes the film before it commits to a label
# rather than writing a rationale to justify a label it has already picked.
EVAL_SYSTEM_PROMPT_V2 = (
    "You are an expert thoracic radiologist reviewing a chest X-ray as part "
    "of an internal evaluation.\n\n"
    "Work in this order: describe what you can actually see on the film, and "
    "only then commit to a single category. Do not name a category before you "
    "have described the evidence for it.\n\n"
    "Call a category ONLY when its defining sign is visible:\n"
    "- pneumothorax: a visceral pleural line, with NO lung markings peripheral "
    "to that line. General lucency or a dark hemithorax is not sufficient.\n"
    "- pleural_effusion: blunting of a costophrenic angle, or a meniscus-shaped "
    "opacity layering at a lung base.\n"
    "- cardiomegaly: cardiothoracic ratio above roughly 0.5 on a PA film.\n"
    "- consolidation: a dense airspace opacity, often with air bronchograms, "
    "that does not layer with gravity.\n"
    "- nodule_or_mass: a discrete rounded opacity with definable borders.\n"
    "- fracture: a cortical break or step in a rib, clavicle, or humerus.\n"
    "- no_finding: none of the above signs is present. This is a legitimate "
    "answer and is preferable to guessing a pathology.\n\n"
    "Output ONLY a single JSON object (no markdown, no prose, no code fences) "
    "with exactly these keys, in this order:\n\n"
    "{\n"
    '  "key_observations": "<2-4 sentences describing the specific features '
    'visible on this image, written BEFORE choosing a category>",\n'
    '  "primary_finding": {\n'
    '    "category": "<one of: pneumothorax, pleural_effusion, cardiomegaly, '
    'consolidation, nodule_or_mass, fracture, no_finding>",\n'
    '    "label": "<short human-readable label>",\n'
    '    "confidence": <number between 0.0 and 1.0>,\n'
    '    "rationale": "<1-3 sentences naming the sign from the list above that '
    'you observed>"\n'
    "  },\n"
    '  "differential_findings": [\n'
    '    {"label": "<differential 1 label>", "confidence": <0.0-1.0>}\n'
    "  ],\n"
    '  "teaching_point": "<1-2 sentence educational takeaway>"\n'
    "}\n\n"
    "Be specific to what you see on the image; do not invent features. Keep "
    "every string short enough that the whole JSON object is complete."
)

EVAL_USER_PROMPT_V2 = (
    "Examine the chest radiograph. Clinical context: {clinical_context}. "
    "Describe what you see in `key_observations` first, then choose the one "
    "category whose defining sign you actually observed. Respond with the "
    "JSON object described above and nothing else."
)


# --- e3.0 -------------------------------------------------------------------
# e2.0 fixed calibration but not accuracy, and bf16 (which restores the model's
# reasoning block, suppressed at 4-bit) did not move it either: 25% at n=12.
# The one category the model gets right is cardiomegaly (2/2) — the only one
# decidable by a *measurement* rather than a texture judgement. e3.0 generalises
# that: it forces a fixed set of observations to be answered for EVERY case
# before any category is named, so the model cannot pick a label and then hunt
# for support. The checklist is answered in full even when the first item is
# positive, which is what distinguishes this from e2.0's per-category sign list.
#
# The `checklist` key is extra output the parser ignores (it reads via .get()),
# but it is recorded in the raw response, so a run can be audited for whether a
# wrong label followed from a wrong observation or from a correct one.
EVAL_SYSTEM_PROMPT_V3 = (
    "You are an expert thoracic radiologist reviewing a chest X-ray as part "
    "of an internal evaluation.\n\n"
    "Complete a fixed inspection checklist BEFORE naming any diagnosis. "
    "Answer every item, in order, even after you have found something "
    "abnormal — a film may show several findings, and skipping items is how "
    "findings get missed. Do not let an early impression change a later "
    "answer.\n\n"
    "Checklist (answer each with what you actually see):\n"
    "1. heart: estimate the cardiothoracic ratio as a decimal (e.g. 0.45).\n"
    "2. costophrenic_angles: 'sharp', 'blunted left', 'blunted right', or "
    "'blunted both'.\n"
    "3. pleural_line: is there a visceral pleural line with NO lung markings "
    "peripheral to it? 'none', 'left', or 'right'.\n"
    "4. focal_opacity: a discrete rounded opacity with definable borders? "
    "'none', or a short location.\n"
    "5. airspace_opacity: a dense non-gravitational opacity, possibly with air "
    "bronchograms? 'none', or a short location.\n"
    "6. bones: a cortical break or step in a rib, clavicle, or humerus? "
    "'intact', or a short location.\n\n"
    "Then choose the category that follows from the checklist:\n"
    "- ratio above roughly 0.5 -> cardiomegaly\n"
    "- blunted costophrenic angle -> pleural_effusion\n"
    "- pleural line present -> pneumothorax\n"
    "- focal opacity present -> nodule_or_mass\n"
    "- airspace opacity present -> consolidation\n"
    "- cortical break present -> fracture\n"
    "- every item normal -> no_finding. This is a legitimate answer and is "
    "preferable to guessing a pathology.\n\n"
    "If several items are positive, make the most prominent one the primary "
    "finding and put the others in differential_findings.\n\n"
    "Output ONLY a single JSON object (no markdown, no prose, no code fences) "
    "with exactly these keys, in this order:\n\n"
    "{\n"
    '  "checklist": {\n'
    '    "heart": "<ratio as a decimal>",\n'
    '    "costophrenic_angles": "<answer>",\n'
    '    "pleural_line": "<answer>",\n'
    '    "focal_opacity": "<answer>",\n'
    '    "airspace_opacity": "<answer>",\n'
    '    "bones": "<answer>"\n'
    "  },\n"
    '  "primary_finding": {\n'
    '    "category": "<one of: pneumothorax, pleural_effusion, cardiomegaly, '
    'consolidation, nodule_or_mass, fracture, no_finding>",\n'
    '    "label": "<short human-readable label>",\n'
    '    "confidence": <number between 0.0 and 1.0>,\n'
    '    "rationale": "<1-2 sentences naming which checklist item led here>"\n'
    "  },\n"
    '  "differential_findings": [\n'
    '    {"label": "<differential 1 label>", "confidence": <0.0-1.0>}\n'
    "  ],\n"
    '  "key_observations": "<2-3 sentences summarising the positive checklist '
    'items>",\n'
    '  "teaching_point": "<1-2 sentence educational takeaway>"\n'
    "}\n\n"
    "Keep every checklist answer to a few words. Be specific to what you see "
    "on the image; do not invent features. Keep every string short enough that "
    "the whole JSON object is complete."
)

EVAL_USER_PROMPT_V3 = (
    "Examine the chest radiograph. Clinical context: {clinical_context}. "
    "Work through all six checklist items first, then let the category follow "
    "from what you recorded. Respond with the JSON object described above and "
    "nothing else."
)


# --- e3.1 -------------------------------------------------------------------
# e3.0 scored 16.7%, below e1.0's 25%, and collapsed onto pleural_effusion on
# 9 of 12 cases. A raw probe showed the checklist itself was not the problem:
# on a cardiomegaly case the model recorded heart "0.55" — above the 0.5
# threshold the prompt itself gave it — and then chose pleural_effusion anyway,
# demoting cardiomegaly to a differential. Three defects in e3.0, all fixable:
#
# 1. Arbitration was undefined. "Make the most prominent one primary" left the
#    model to invent a tie-break, and it always picked the same one. e3.1
#    states the priority order explicitly. That order is the one
#    `find_primary_category` uses to build the ground truth, so this tells the
#    model the labelling convention it is being scored against rather than
#    making it guess ours. This must be declared in the write-up: it aligns the
#    task definition with the metric, and a run under e3.1 is not comparable to
#    a hypothetical model that had to infer the convention.
# 2. Forcing a verdict on all six items manufactured false positives ("blunted
#    left" on a film NIH labels cardiomegaly-only). e3.1 makes "normal" the
#    default and demands an unambiguous sign before any positive.
# 3. e3.0 halved latency (~75s to ~30s), i.e. the rigid form displaced the
#    model's own reasoning block, which e1.0 did produce. e3.1 explicitly asks
#    it to reason before filling anything in.
EVAL_SYSTEM_PROMPT_V3_1 = (
    "You are an expert thoracic radiologist reviewing a chest X-ray as part "
    "of an internal evaluation.\n\n"
    "Take your time and reason about the film before you write anything. Then "
    "complete a fixed inspection checklist, and only then name a diagnosis.\n\n"
    "Checklist. Record the NORMAL answer unless the abnormal sign is "
    "unambiguous — most films are normal for most of these items, and a "
    "hedged or borderline appearance is a normal answer:\n"
    "1. heart: cardiothoracic ratio as a decimal (e.g. 0.45).\n"
    "2. costophrenic_angles: 'sharp' unless an angle is clearly blunted or a "
    "meniscus is visible; then 'blunted left' / 'blunted right' / "
    "'blunted both'.\n"
    "3. pleural_line: 'none' unless you can see a visceral pleural line WITH "
    "no lung markings peripheral to it; then 'left' or 'right'. A dark or "
    "lucent hemithorax alone is 'none'.\n"
    "4. focal_opacity: 'none' unless there is a discrete rounded opacity with "
    "definable borders; then a short location.\n"
    "5. airspace_opacity: 'none' unless there is a dense non-gravitational "
    "opacity, often with air bronchograms; then a short location.\n"
    "6. bones: 'intact' unless there is a cortical break or step.\n\n"
    "Each item maps to one category:\n"
    "- ratio above 0.5 -> cardiomegaly\n"
    "- blunted angle -> pleural_effusion\n"
    "- pleural line -> pneumothorax\n"
    "- focal opacity -> nodule_or_mass\n"
    "- airspace opacity -> consolidation\n"
    "- cortical break -> fracture\n\n"
    "Choosing the primary finding — follow these rules exactly:\n"
    "- If every item is normal, the category is no_finding. This is a "
    "legitimate answer and is preferable to guessing a pathology.\n"
    "- If exactly one item is positive, its category is the primary finding.\n"
    "- If several items are positive, pick the primary finding by this fixed "
    "priority, highest first: pneumothorax, pleural_effusion, cardiomegaly, "
    "consolidation, fracture, nodule_or_mass. Put the remaining positive "
    "categories in differential_findings.\n"
    "- The primary finding MUST be a category whose checklist item you "
    "recorded as positive. Do not name a category you did not observe, and do "
    "not record a positive item and then leave its category out entirely.\n\n"
    "Output ONLY a single JSON object (no markdown, no prose, no code fences) "
    "with exactly these keys, in this order:\n\n"
    "{\n"
    '  "checklist": {\n'
    '    "heart": "<ratio as a decimal>",\n'
    '    "costophrenic_angles": "<answer>",\n'
    '    "pleural_line": "<answer>",\n'
    '    "focal_opacity": "<answer>",\n'
    '    "airspace_opacity": "<answer>",\n'
    '    "bones": "<answer>"\n'
    "  },\n"
    '  "positive_items": ["<names of the checklist items that were abnormal, '
    'or an empty list>"],\n'
    '  "primary_finding": {\n'
    '    "category": "<one of: pneumothorax, pleural_effusion, cardiomegaly, '
    'consolidation, nodule_or_mass, fracture, no_finding>",\n'
    '    "label": "<short human-readable label>",\n'
    '    "confidence": <number between 0.0 and 1.0>,\n'
    '    "rationale": "<1-2 sentences naming the checklist item and the '
    'priority rule that selected this category>"\n'
    "  },\n"
    '  "differential_findings": [\n'
    '    {"label": "<differential 1 label>", "confidence": <0.0-1.0>}\n'
    "  ],\n"
    '  "key_observations": "<2-3 sentences summarising the positive checklist '
    'items>",\n'
    '  "teaching_point": "<1-2 sentence educational takeaway>"\n'
    "}\n\n"
    "Keep every checklist answer to a few words. Be specific to what you see "
    "on the image; do not invent features. Keep every string short enough that "
    "the whole JSON object is complete."
)

EVAL_USER_PROMPT_V3_1 = (
    "Examine the chest radiograph. Clinical context: {clinical_context}. "
    "Reason about the film first, then fill in all six checklist items, "
    "recording the normal answer wherever the abnormal sign is not "
    "unambiguous. List which items were positive, then apply the priority "
    "rule to pick the primary finding. Respond with the JSON object described "
    "above and nothing else."
)


# --- e4.0 -------------------------------------------------------------------
# e1.0 (loose schema) ran ~75s/case and produced MedGemma's reasoning block.
# e3.0 and e3.1 (rigid schema) ran ~31s/case with no reasoning block at all,
# and both collapsed onto a single category. The consistent signal across four
# variants is that tightening the output structure suppresses deliberation, so
# e4.0 goes the other way: dictate a free-text report FIRST, in the model's own
# words and with no schema in view, and only then emit the JSON.
#
# The prose is not decoration — it is the reasoning, made explicit and, unlike
# the <unused94> block, actually visible in the response text. The parser hunts
# for JSON inside surrounding prose already, so nothing downstream changes.
#
# Note this cannot fully decouple seeing from labelling: the category list is
# still in the system prompt while the prose is written. A two-stage variant
# (describe in one call, classify the description in a second) is the version
# that would remove that, at twice the GPU cost.
EVAL_SYSTEM_PROMPT_V4 = (
    "You are an expert thoracic radiologist reviewing a chest X-ray as part "
    "of an internal evaluation.\n\n"
    "Answer in two parts.\n\n"
    "PART 1 — Dictate your report as prose, the way you would speak it to a "
    "colleague. No JSON, no bullet points, no headings. Work through the film "
    "systematically: lung fields, pleural surfaces, mediastinum and heart "
    "size, diaphragm and costophrenic angles, bones and soft tissues. Say what "
    "you actually see, including what looks normal. Weigh the possibilities "
    "out loud, including the ones you end up rejecting and why. Take as long "
    "as you need here; this part matters more than the second.\n\n"
    "PART 2 — After the prose, on a new line, output a single JSON object "
    "(no markdown, no code fences) that records the conclusion your report "
    "reached:\n\n"
    "{\n"
    '  "primary_finding": {\n'
    '    "category": "<one of: pneumothorax, pleural_effusion, cardiomegaly, '
    'consolidation, nodule_or_mass, fracture, no_finding>",\n'
    '    "label": "<short human-readable label>",\n'
    '    "confidence": <number between 0.0 and 1.0>,\n'
    '    "rationale": "<1-2 sentences, drawn from your report above>"\n'
    "  },\n"
    '  "differential_findings": [\n'
    '    {"label": "<differential 1 label>", "confidence": <0.0-1.0>}\n'
    "  ],\n"
    '  "key_observations": "<2-3 sentences summarising the findings from your '
    'report>",\n'
    '  "teaching_point": "<1-2 sentence educational takeaway>"\n'
    "}\n\n"
    "The JSON must follow from the report, not replace it: do not name a "
    "category in Part 2 that your prose did not describe evidence for. If the "
    "film is unremarkable, `no_finding` is the correct answer and is "
    "preferable to guessing a pathology. Do not invent features."
)

EVAL_USER_PROMPT_V4 = (
    "Examine the chest radiograph. Clinical context: {clinical_context}. "
    "Dictate your radiology report as prose first, then give the JSON object "
    "recording what you concluded."
)


# --- e5.0 -------------------------------------------------------------------
# Two-stage. e4.0 restored the model's prose reasoning and broke the
# single-category collapse, but accuracy did not move, and one confound
# remained: the seven category names sit in the system prompt while the prose
# is being written, so "describing the film" and "choosing a label" still
# happen in one pass with the answer options in view.
#
# e5.0 separates them. Stage 1 sees no category list at all and is told not to
# diagnose — it only reports what is on the film. Stage 2 receives that text
# and assigns the category. If accuracy is limited by the model's visual
# discrimination, this changes nothing; if it is limited by premature
# commitment to a label, this is the variant that should show it.
#
# Honest limitation: the port requires an image on every call, so stage 2 still
# receives the radiograph and could re-derive a label from pixels rather than
# from the report. The prompt tells it not to, which is weaker than removing
# the image. A text-only classifier call would need a port change.
EVAL_DESCRIBE_SYSTEM_V5 = (
    "You are an expert thoracic radiologist dictating the findings section of "
    "a report.\n\n"
    "Describe this chest X-ray as prose, the way you would speak it to a "
    "colleague. Work through the film systematically: lung fields, pleural "
    "surfaces, mediastinum and heart size, diaphragm and costophrenic angles, "
    "bones and soft tissues, and any lines or devices. Give the "
    "cardiothoracic ratio as a decimal. Say what you actually see, including "
    "what looks normal, and be specific about side and location.\n\n"
    "Do NOT give a diagnosis, do not name a disease, and do not state an "
    "impression or conclusion. Findings only. Another radiologist will read "
    "your description and draw the conclusion, so it must stand on its own: "
    "anything you leave out, they cannot see.\n\n"
    "Output the prose only. No JSON, no headings, no bullet points."
)

EVAL_DESCRIBE_USER_V5 = (
    "Examine the chest radiograph. Clinical context: {clinical_context}. "
    "Dictate the findings section only — no diagnosis, no impression."
)

EVAL_SYSTEM_PROMPT_V5 = (
    "You are an expert thoracic radiologist. A colleague has examined a chest "
    "X-ray and dictated their findings. Your job is to read their description "
    "and assign the single category it best supports.\n\n"
    "Base your answer on the description. Do not introduce findings your "
    "colleague did not report.\n\n"
    "Categories, and what the description must contain to support each:\n"
    "- pneumothorax: a visceral pleural line with no lung markings beyond it.\n"
    "- pleural_effusion: a blunted costophrenic angle, or a meniscus at a base.\n"
    "- cardiomegaly: a cardiothoracic ratio above roughly 0.5.\n"
    "- consolidation: dense airspace opacity, possibly with air bronchograms.\n"
    "- nodule_or_mass: a discrete rounded opacity with definable borders.\n"
    "- fracture: a cortical break or step in a bone.\n"
    "- no_finding: the description reports no abnormality of the above kinds. "
    "This is a legitimate answer and is preferable to guessing a pathology.\n\n"
    "If the description supports several, make the most prominent one primary "
    "and put the rest in differential_findings.\n\n"
    "Output ONLY a single JSON object (no markdown, no code fences):\n\n"
    "{\n"
    '  "primary_finding": {\n'
    '    "category": "<one of: pneumothorax, pleural_effusion, cardiomegaly, '
    'consolidation, nodule_or_mass, fracture, no_finding>",\n'
    '    "label": "<short human-readable label>",\n'
    '    "confidence": <number between 0.0 and 1.0>,\n'
    '    "rationale": "<1-2 sentences quoting what in the description '
    'supports this>"\n'
    "  },\n"
    '  "differential_findings": [\n'
    '    {"label": "<differential 1 label>", "confidence": <0.0-1.0>}\n'
    "  ],\n"
    '  "key_observations": "<2-3 sentences summarising the description>",\n'
    '  "teaching_point": "<1-2 sentence educational takeaway>"\n'
    "}"
)

EVAL_USER_PROMPT_V5 = (
    "Clinical context: {clinical_context}.\n\n"
    "Your colleague's dictated findings:\n"
    '"""\n'
    "{report}\n"
    '"""\n\n'
    "Assign the category their description best supports, judging by the "
    "description rather than by re-reading the image yourself. Respond with "
    "the JSON object described above and nothing else."
)


@dataclass(frozen=True, slots=True)
class EvalPromptSet:
    """A versioned pair of eval prompts.

    The harness uses its own bias-free prompts, not `PromptBuilder`'s, so it
    carries its own versions. Bump `version` on any deliberate wording change;
    the fingerprint catches an edit that forgets to.

    `describe_system` / `describe_user` are optional. When both are set the
    harness makes an extra call FIRST to obtain a free-text description, then
    interpolates it into `user` as `{report}` for the classifying call. A set
    without them behaves exactly as before, so every existing version is
    unaffected.
    """

    version: str
    system: str
    user: str
    describe_system: str | None = None
    describe_user: str | None = None

    def fingerprint(self) -> str:
        parts = [self.system, self.user]
        if self.is_two_stage:
            # Stage-1 wording changes the stage-2 input and therefore the
            # result, so it has to be inside the fingerprint.
            parts += [self.describe_system or "", self.describe_user or ""]
        joined = "\n--- user ---\n".join(parts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]

    def qualified_version(self) -> str:
        return f"{self.version}+{self.fingerprint()}"

    @property
    def is_two_stage(self) -> bool:
        """True when the set describes the film in a separate first call.

        Both stage-1 fields must be set; one alone is a mistake rather than a
        meaningful configuration, and silently ignoring it would produce a
        report labelled two-stage that ran as one.
        """
        has_system = self.describe_system is not None
        has_user = self.describe_user is not None
        if has_system != has_user:
            raise ValidationError(
                f"Prompt set {self.version!r} sets only one of describe_system/"
                "describe_user. A two-stage set needs both."
            )
        return has_system


EVAL_PROMPT_VERSION: Final[str] = "e1.0"
DEFAULT_EVAL_PROMPT_VERSION: Final[str] = EVAL_PROMPT_VERSION

EVAL_PROMPT_SETS: Final[dict[str, EvalPromptSet]] = {
    "e1.0": EvalPromptSet("e1.0", EVAL_SYSTEM_PROMPT, EVAL_USER_PROMPT),
    "e2.0": EvalPromptSet("e2.0", EVAL_SYSTEM_PROMPT_V2, EVAL_USER_PROMPT_V2),
    "e3.0": EvalPromptSet("e3.0", EVAL_SYSTEM_PROMPT_V3, EVAL_USER_PROMPT_V3),
    "e3.1": EvalPromptSet("e3.1", EVAL_SYSTEM_PROMPT_V3_1, EVAL_USER_PROMPT_V3_1),
    "e4.0": EvalPromptSet("e4.0", EVAL_SYSTEM_PROMPT_V4, EVAL_USER_PROMPT_V4),
    "e5.0": EvalPromptSet(
        "e5.0",
        EVAL_SYSTEM_PROMPT_V5,
        EVAL_USER_PROMPT_V5,
        describe_system=EVAL_DESCRIBE_SYSTEM_V5,
        describe_user=EVAL_DESCRIBE_USER_V5,
    ),
}


def resolve_prompt_set(version: str | None = None) -> EvalPromptSet:
    """Look up a prompt set by version, raising on an unknown one.

    An unknown version must not silently fall back to the default: the whole
    point of versioning is that a report names the prompt that produced it.
    """
    key = version or DEFAULT_EVAL_PROMPT_VERSION
    try:
        return EVAL_PROMPT_SETS[key]
    except KeyError:
        known = ", ".join(sorted(EVAL_PROMPT_SETS))
        raise ValidationError(f"Unknown eval prompt version {key!r}. Known: {known}.") from None


def eval_prompt_fingerprint() -> str:
    joined = "\n--- user ---\n".join((EVAL_SYSTEM_PROMPT, EVAL_USER_PROMPT))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def eval_prompt_version() -> str:
    return f"{EVAL_PROMPT_VERSION}+{eval_prompt_fingerprint()}"


__all__ = [
    "EvalSample",
    "EvalReport",
    "EvalPromptSet",
    "EVAL_SYSTEM_PROMPT",
    "EVAL_USER_PROMPT",
    "EVAL_SYSTEM_PROMPT_V2",
    "EVAL_USER_PROMPT_V2",
    "EVAL_SYSTEM_PROMPT_V3",
    "EVAL_USER_PROMPT_V3",
    "EVAL_SYSTEM_PROMPT_V3_1",
    "EVAL_USER_PROMPT_V3_1",
    "EVAL_SYSTEM_PROMPT_V4",
    "EVAL_USER_PROMPT_V4",
    "EVAL_SYSTEM_PROMPT_V5",
    "EVAL_USER_PROMPT_V5",
    "EVAL_DESCRIBE_SYSTEM_V5",
    "EVAL_DESCRIBE_USER_V5",
    "EVAL_PROMPT_VERSION",
    "EVAL_PROMPT_SETS",
    "DEFAULT_EVAL_PROMPT_VERSION",
    "resolve_prompt_set",
    "eval_prompt_fingerprint",
    "eval_prompt_version",
    "select_eval_cases",
    "run_evaluation",
    "write_json_report",
    "write_markdown_summary",
    "render_markdown_summary",
]


@dataclass(frozen=True, slots=True)
class EvalSample:
    case_id: str
    ground_truth_category: str
    predicted_category: str | None
    parse_succeeded: bool
    latency_ms: float
    prompt_tokens: int
    output_tokens: int
    error: str | None = None
    model: str = ""
    # Every category the case's NIH labels support, not just the primary one
    # `find_primary_category` reduced them to. 43% of the corpus is multi-label,
    # so a prediction can be genuinely present on the film and still miss the
    # primary category; `lenient_correct` counts those as hits.
    all_true_categories: tuple[str, ...] = ()
    lenient_correct: bool = False


@dataclass(frozen=True, slots=True)
class EvalReport:
    runtime: str
    total_cases: int
    parse_succeeded: int
    parse_failed: int
    structure_validity: float
    finding_recognition: float
    per_finding_recognition: dict[str, dict[str, float]]
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    total_prompt_tokens: int
    total_output_tokens: int
    # Recognition scored against every category the NIH labels support. It is
    # an upper bound, not a better number: on a four-label case, naming a
    # common finding scores correct by luck. Reported alongside the strict
    # figure so the gap — which is a property of our label reduction, not of
    # the model — is visible in every report rather than only in an audit.
    finding_recognition_lenient: float = 0.0
    # Provenance: which model, which prompt and which sample produced these
    # numbers. Defaulted so reports written before these fields existed still load.
    model: str = ""
    prompt_version: str = ""
    split: str = ""
    sampling: str = ""
    samples: tuple[EvalSample, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _clinical_context(case: Case) -> str:
    return case.metadata.get("clinical_context", "").strip() or "(no clinical context provided)"


def _build_evaluation_prompt(
    case: Case, prompts: EvalPromptSet, *, report: str = ""
) -> tuple[str, str]:
    """Render the classifying prompt.

    `report` is the stage-1 description for a two-stage set and "" otherwise.
    It is passed as a `.format` *value*, never as part of the template, so
    braces in model-generated prose cannot blow up the interpolation.
    """
    return prompts.system, prompts.user.format(
        clinical_context=_clinical_context(case), report=report
    )


def select_eval_cases(
    case_repository: ICaseRepository,
    *,
    split: str | None,
    max_cases: int,
    stratified: bool = True,
) -> list[Case]:
    """Choose the cases to evaluate.

    Taking the first `max_cases` manifest rows silently inherits whatever
    order the manifest happens to be in. On the curated NIH subset that gave a
    30-case run with 11 cardiomegaly, 3 consolidation and no negatives at all,
    which makes the per-finding table unreadable — a "0.0%" that is really 0/3.

    Stratified selection round-robins across the ground-truth categories
    present in the split, so every category contributes a comparable number of
    cases and a category with fewer rows simply drops out of the rotation once
    exhausted. Selection stays deterministic (manifest order within each
    category) so two runs are comparable.

    Within a category, a case belonging to a patient already chosen is skipped
    on the first pass. Category stratification alone is blind to patients, and
    the curated subset is 300 images from 86 patients: a 12-case sample drew
    5 films of one man, whose pacemaker and chest tube then recurred across
    the run and read as model behaviour. Deferred cases are appended once the
    unseen-patient supply is exhausted, so `max_cases` is still filled — the
    pass only changes the *order* in which duplicates are reached, which is
    what matters when the cap truncates the list.
    """
    available = case_repository.list(split=split, limit=None)
    if not stratified or not available:
        return available[:max_cases]

    by_category: dict[str, list[Case]] = {}
    for case in available:
        by_category.setdefault(case.ground_truth_finding.category.value, []).append(case)

    chosen: list[Case] = []
    deferred: list[Case] = []
    seen_patients: set[str] = set()
    queues = [by_category[key] for key in sorted(by_category)]
    position = 0
    while any(len(q) > position for q in queues):
        for queue in queues:
            if len(queue) <= position:
                continue
            case = queue[position]
            patient = case.patient_id
            # An empty patient id means the case carries no image index — a
            # hand-built or synthetic case. Those are never treated as
            # duplicates of each other.
            if patient and patient in seen_patients:
                deferred.append(case)
                continue
            if patient:
                seen_patients.add(patient)
            chosen.append(case)
        position += 1

    chosen.extend(deferred)
    return chosen[:max_cases]


def run_evaluation(
    *,
    case_repository: ICaseRepository,
    analyzer: IImageAnalyzer,
    parser: OutputParser,
    max_cases: int,
    runtime_label: str,
    split: str | None = "test",
    image_loader: IImageLoader | None = None,
    prompts: EvalPromptSet | None = None,
    stratified: bool = True,
) -> EvalReport:
    """Run the eval harness over the `eval` (or specified) split.

    Caps at `max_cases`; if fewer are available, runs over what is
    available. The dry-run analyzer is fast enough that the cap is
    mostly a guard against Kaggle-quota burn.
    """
    loader = image_loader or ByReferenceImageLoader()
    prompt_set = prompts or resolve_prompt_set()
    sampling = "stratified" if stratified else "manifest_order"
    cases = select_eval_cases(
        case_repository, split=split, max_cases=max_cases, stratified=stratified
    )
    if not cases:
        logger.warning("No cases found in split=%r; producing an empty report.", split)
        return _empty_report(runtime_label, split or "", prompt_set, sampling)

    logger.info(
        "eval sample (%s, n=%d): %s",
        sampling,
        len(cases),
        dict(sorted(Counter(c.ground_truth_finding.category.value for c in cases).items())),
    )

    samples: list[EvalSample] = []
    for index, case in enumerate(cases, start=1):
        ground = case.ground_truth_finding.category.value
        attempt_id = f"eval-{index:03d}-{case.id}"
        sample = _evaluate_one(
            case=case,
            analyzer=analyzer,
            parser=parser,
            prompts=prompt_set,
            attempt_id=attempt_id,
            image_loader=loader,
        )
        samples.append(sample)
        logger.info(
            "eval %d/%d case=%s gt=%s pred=%s parsed=%s latency=%.0fms",
            index,
            len(cases),
            case.id,
            ground,
            sample.predicted_category or "<unparsed>",
            sample.parse_succeeded,
            sample.latency_ms,
        )

    return _build_report(runtime_label, tuple(samples), split or "", prompt_set, sampling)


def _evaluate_one(
    *,
    case: Case,
    analyzer: IImageAnalyzer,
    parser: OutputParser,
    prompts: EvalPromptSet,
    attempt_id: str,
    image_loader: IImageLoader,
) -> EvalSample:
    started = time.perf_counter()
    error: str | None = None
    parse_succeeded = False
    predicted: str | None = None
    prompt_tokens = 0
    output_tokens = 0
    model = ""
    context = _clinical_context(case)

    try:
        image = image_loader.load(case.image_url)
        report_text = ""
        if prompts.is_two_stage:
            # Stage 1: findings only, no category list in view. Its tokens and
            # latency belong to this case, so they are accumulated rather than
            # measured separately — a two-stage run genuinely costs two calls.
            described = analyzer.analyze(
                image=image,
                prompt=(prompts.describe_user or "").format(clinical_context=context),
                system=prompts.describe_system or "",
            )
            prompt_tokens += int(described.model_output.prompt_tokens)
            output_tokens += int(described.model_output.output_tokens)
            model = described.model_output.model
            report_text = described.raw_text.strip()
            if not report_text:
                raise UpstreamUnavailableError("Stage 1 returned an empty description.")

        system, user = _build_evaluation_prompt(case, prompts, report=report_text)
        analysis = analyzer.analyze(image=image, prompt=user, system=system)
        prompt_tokens += int(analysis.model_output.prompt_tokens)
        output_tokens += int(analysis.model_output.output_tokens)
        model = analysis.model_output.model
        try:
            feedback = parser.parse(attempt_id=attempt_id, raw_text=analysis.raw_text)
            parse_succeeded = True
            predicted = _extract_category(feedback)
        except (ValidationError, UpstreamUnavailableError) as exc:
            error = f"parse: {exc.__class__.__name__}"
    except (UpstreamUnavailableError, ValidationError) as exc:
        error = f"analyze: {exc.__class__.__name__}"
    except Exception as exc:  # defensive: never let one bad case kill the run
        error = f"unexpected: {exc.__class__.__name__}"
        logger.exception("Unexpected error evaluating case %s", case.id)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    all_true = _all_true_categories(case)
    return EvalSample(
        case_id=case.id,
        ground_truth_category=case.ground_truth_finding.category.value,
        predicted_category=predicted,
        parse_succeeded=parse_succeeded,
        latency_ms=elapsed_ms,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        error=error,
        model=model,
        all_true_categories=all_true,
        lenient_correct=predicted is not None and predicted in all_true,
    )


def _all_true_categories(case: Case) -> tuple[str, ...]:
    """Every one of our categories the case's NIH labels support.

    Falls back to the primary category when a case carries no `nih_labels`
    metadata — a hand-built or synthetic case, where the primary label is all
    the truth there is. Lenient scoring then degrades to strict for that case,
    which is the safe direction.
    """
    primary = case.ground_truth_finding.category.value
    raw = case.metadata.get("nih_labels")
    if isinstance(raw, str):
        labels: list[str] = [raw]
    elif isinstance(raw, list | tuple):
        labels = [str(item) for item in raw]
    else:
        return (primary,)
    if not labels:
        return (primary,)
    categories = {c.value for c in nih_labels_to_finding_categories(labels)}
    categories.add(primary)
    return tuple(sorted(categories))


def _extract_category(feedback) -> str:  # type: ignore[no-untyped-def]
    """Pull the canonical category out of a parsed Feedback.

    The `OutputParser` stores the canonicalised category under
    `raw["parsed"]["primary_finding"]["category"]`; the public
    `feedback.likely_finding` carries the human-readable label which
    can be any free-form text. For accuracy metrics we want the
    category, not the label.
    """
    raw = feedback.raw if isinstance(feedback.raw, dict) else {}
    parsed = raw.get("parsed") if isinstance(raw, dict) else None
    if isinstance(parsed, dict):
        primary = parsed.get("primary_finding")
        if isinstance(primary, dict):
            cat = primary.get("category")
            if isinstance(cat, str) and cat:
                return cat
    return str(feedback.likely_finding.value)


def _reported_model(samples: tuple[EvalSample, ...]) -> str:
    """The model the server reported. Warn rather than guess if it varied."""
    seen = {s.model for s in samples if s.model}
    if len(seen) > 1:
        logger.warning("Samples span more than one model: %s", sorted(seen))
        return ",".join(sorted(seen))
    return next(iter(seen), "")


def _build_report(
    runtime_label: str,
    samples: tuple[EvalSample, ...],
    split: str = "",
    prompts: EvalPromptSet | None = None,
    sampling: str = "",
) -> EvalReport:
    total = len(samples)
    parsed = [s for s in samples if s.parse_succeeded]
    correct = [s for s in parsed if s.predicted_category == s.ground_truth_category]
    lenient = [s for s in parsed if s.lenient_correct]

    per_finding_total: Counter[str] = Counter(s.ground_truth_category for s in samples)
    per_finding_correct: Counter[str] = Counter(s.ground_truth_category for s in correct)

    per_finding: dict[str, dict[str, float]] = {}
    for category in sorted(per_finding_total):
        denom = per_finding_total[category]
        numer = per_finding_correct[category]
        per_finding[category] = {
            "total": float(denom),
            "correct": float(numer),
            "accuracy": (numer / denom) if denom else 0.0,
        }

    latencies = [s.latency_ms for s in samples]
    total_prompt = sum(s.prompt_tokens for s in samples)
    total_output = sum(s.output_tokens for s in samples)

    return EvalReport(
        runtime=runtime_label,
        total_cases=total,
        parse_succeeded=len(parsed),
        parse_failed=total - len(parsed),
        structure_validity=(len(parsed) / total) if total else 0.0,
        finding_recognition=(len(correct) / total) if total else 0.0,
        per_finding_recognition=per_finding,
        finding_recognition_lenient=(len(lenient) / total) if total else 0.0,
        latency_mean_ms=statistics.fmean(latencies) if latencies else 0.0,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        total_prompt_tokens=total_prompt,
        total_output_tokens=total_output,
        model=_reported_model(samples),
        prompt_version=(prompts.qualified_version() if prompts else eval_prompt_version()),
        split=split,
        sampling=sampling,
        samples=samples,
    )


def _empty_report(
    runtime_label: str,
    split: str = "",
    prompts: EvalPromptSet | None = None,
    sampling: str = "",
) -> EvalReport:
    return EvalReport(
        runtime=runtime_label,
        total_cases=0,
        parse_succeeded=0,
        parse_failed=0,
        structure_validity=0.0,
        finding_recognition=0.0,
        per_finding_recognition={},
        latency_mean_ms=0.0,
        latency_p50_ms=0.0,
        latency_p95_ms=0.0,
        total_prompt_tokens=0,
        total_output_tokens=0,
        model="",
        prompt_version=(prompts.qualified_version() if prompts else eval_prompt_version()),
        split=split,
        sampling=sampling,
        samples=(),
    )


def write_json_report(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_markdown_summary(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_summary(report), encoding="utf-8")


def render_markdown_summary(report: EvalReport) -> str:
    lines: list[str] = []
    lines.append("# Phase 1 Evaluation Report")
    lines.append("")
    lines.append(f"- Runtime: `{report.runtime}`")
    lines.append(f"- Model: `{report.model or 'unknown'}`")
    lines.append(f"- Prompt version: `{report.prompt_version or 'unknown'}`")
    if report.split:
        lines.append(f"- Split: `{report.split}`")
    if report.sampling:
        lines.append(f"- Sampling: `{report.sampling}`")
    lines.append(f"- Total cases: {report.total_cases}")
    lines.append(f"- Parsed: {report.parse_succeeded}")
    lines.append(f"- Parse failures: {report.parse_failed}")
    lines.append(
        f"- **Structure validity:** {report.structure_validity * 100:.1f}%"
    )
    lines.append(
        f"- **Finding recognition (strict):** {report.finding_recognition * 100:.1f}%"
    )
    lines.append(
        f"- **Finding recognition (lenient):** "
        f"{report.finding_recognition_lenient * 100:.1f}% "
        "— credits any category the case's NIH labels support"
    )
    lines.append(
        f"- Latency: mean={report.latency_mean_ms:.0f}ms  "
        f"p50={report.latency_p50_ms:.0f}ms  p95={report.latency_p95_ms:.0f}ms"
    )
    lines.append(
        f"- Token usage: {report.total_prompt_tokens} prompt + "
        f"{report.total_output_tokens} output"
    )
    lines.append("")
    lines.append("## Per-finding recognition")
    lines.append("")
    lines.append("| Ground truth | n | correct | accuracy |")
    lines.append("|---|---:|---:|---:|")
    for category, stats in report.per_finding_recognition.items():
        lines.append(
            f"| `{category}` | {int(stats['total'])} | "
            f"{int(stats['correct'])} | {stats['accuracy'] * 100:.1f}% |"
        )
    lines.append("")
    if any(s.error for s in report.samples):
        lines.append("## Errors")
        lines.append("")
        lines.append("| Case | Stage | Error |")
        lines.append("|---|---|---|")
        for s in report.samples:
            if s.error:
                stage, name = s.error.split(": ", 1) if ": " in s.error else ("?", s.error)
                lines.append(f"| `{s.case_id}` | {stage} | {name} |")
        lines.append("")
    lines.append(
        "_NIH labels are NLP-mined radiology reports and are known to be noisy. "
        "Treat per-finding accuracy as a directional signal, not a clinical ground truth._"
    )
    return "\n".join(lines) + "\n"
