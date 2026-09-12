"""OutputParser: turn raw model text into a Feedback entity.

The model is asked to output a single JSON object. Real outputs vary:
they may include prose around the JSON, markdown code fences, trailing
commas, smart quotes, etc. The parser is permissive on input, strict
on output — anything it can't validate becomes an `UpstreamUnavailableError`
so the route returns 503 and the UI shows "Server error" / a retry.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Final

from cxt.domain.entities.feedback import DifferentialFinding, Feedback
from cxt.domain.value_objects.confidence import Confidence
from cxt.domain.value_objects.finding_label import FindingLabel
from cxt.domain.value_objects.rationale import Rationale
from cxt.shared.errors import UpstreamUnavailableError, ValidationError

logger = logging.getLogger(__name__)

_VALID_CATEGORIES: Final[tuple[str, ...]] = (
    "pneumothorax",
    "pleural_effusion",
    "cardiomegaly",
    "consolidation",
    "nodule_or_mass",
    "fracture",
    "no_finding",
)


class OutputParser:
    """Parse model text into a Feedback entity."""

    def parse(self, *, attempt_id: str, raw_text: str) -> Feedback:
        payload = self._extract_json(raw_text)
        return self._build_feedback(attempt_id, raw_text, payload)

    @staticmethod
    def _extract_json(raw_text: str) -> dict[str, Any]:
        cleaned = raw_text.strip()
        if not cleaned:
            raise UpstreamUnavailableError("Model returned empty output.")

        cleaned = _strip_thinking(cleaned)

        fenced = _FENCE_RE.search(cleaned)
        if fenced is not None:
            cleaned = fenced.group(1).strip()
        else:
            # An unterminated fence (```json with no closing fence) is common
            # when generation stops at the token limit.
            open_fence = _OPEN_FENCE_RE.search(cleaned)
            if open_fence is not None:
                cleaned = cleaned[open_fence.end() :].strip()

        for candidate in _repair_candidates(cleaned):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return dict(parsed)

        raise UpstreamUnavailableError(
            "Model output did not contain a parseable JSON object. "
            f"First 200 chars: {raw_text[:200]!r}"
        )

    @staticmethod
    def _build_feedback(attempt_id: str, raw_text: str, payload: dict[str, Any]) -> Feedback:
        if not isinstance(payload, dict):
            raise UpstreamUnavailableError("Model JSON must be an object.")

        primary = payload.get("primary_finding")
        if not isinstance(primary, dict):
            raise UpstreamUnavailableError("primary_finding missing or not an object.")

        category_raw = primary.get("category")
        if not isinstance(category_raw, str):
            raise UpstreamUnavailableError("primary_finding.category must be a string.")
        category_value = _CANONICALISE_CATEGORY.get(category_raw.strip().lower(), category_raw.strip().lower())
        if category_value not in _VALID_CATEGORIES:
            raise UpstreamUnavailableError(
                f"primary_finding.category {category_raw!r} is not one of {_VALID_CATEGORIES}."
            )

        label_raw = primary.get("label")
        if not isinstance(label_raw, str) or not label_raw.strip():
            label_raw = category_value.replace("_", " ").title()

        confidence_raw = primary.get("confidence")
        if not isinstance(confidence_raw, int | float):
            raise UpstreamUnavailableError("primary_finding.confidence must be a number.")
        confidence = Confidence(float(confidence_raw))

        rationale_raw = primary.get("rationale", "")
        if not isinstance(rationale_raw, str) or not rationale_raw.strip():
            rationale_raw = "No rationale provided by model."
        try:
            rationale = Rationale(rationale_raw)
        except ValidationError:
            rationale = Rationale(rationale_raw[:500] or "Model rationale was empty.")

        key_observations_raw = payload.get("key_observations", "")
        if not isinstance(key_observations_raw, str) or not key_observations_raw.strip():
            key_observations_raw = "(no key observations returned)"
        try:
            key_observations = Rationale(key_observations_raw)
        except ValidationError:
            key_observations = Rationale(key_observations_raw[:500] or "(truncated)")

        teaching_point_raw = payload.get("teaching_point", "")
        if not isinstance(teaching_point_raw, str) or not teaching_point_raw.strip():
            teaching_point_raw = "(no teaching point returned)"
        try:
            teaching_point = Rationale(teaching_point_raw)
        except ValidationError:
            teaching_point = Rationale(teaching_point_raw[:500] or "(truncated)")

        likely_finding = FindingLabel(label_raw.strip()[:200])

        comparison_text = _format_primary_for_comparison(category_value, confidence_raw, rationale_raw)
        differentials = _parse_differentials(
            payload.get("differential_findings"), ceiling=confidence.value
        )
        missed_clues_text = _format_missed_clues(differentials)

        return Feedback(
            attempt_id=attempt_id,
            likely_finding=likely_finding,
            key_signs=rationale,
            comparison=Rationale(comparison_text),
            missed_clues=Rationale(missed_clues_text),
            learning_point=teaching_point,
            confidence=confidence,
            key_observations=key_observations,
            differentials=differentials,
            sources=tuple(),
            raw={"raw_text": raw_text, "parsed": payload, "category": category_value},
        )


_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.IGNORECASE | re.DOTALL,
)
_OPEN_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"```(?:json)?\s*", re.IGNORECASE)
# MedGemma 1.5 reasons before answering, wrapping the reasoning in Gemma's
# private-use tokens: "<unused94>thought ... <unused95>" then the real answer.
# The reasoning is prose full of quoted fragments and category names, so it has
# to be removed before any JSON hunting rather than merely skipped over.
_THINKING_END_RE: Final[re.Pattern[str]] = re.compile(r"<unused9[0-9]>")
_THINKING_START_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*<unused9[0-9]>\s*thought\b", re.IGNORECASE
)
_OBJECT_RE: Final[re.Pattern[str]] = re.compile(r"\{.*\}", re.DOTALL)
_TRAILING_COMMA_RE: Final[re.Pattern[str]] = re.compile(r",\s*([}\]])")

# Models emit typographic quotes when they slip out of JSON mode. Written as
# escapes because the literal glyphs are themselves ambiguous in source.
_SMART_QUOTES: Final[dict[int, str]] = {
    0x201C: '"',  # LEFT DOUBLE QUOTATION MARK
    0x201D: '"',  # RIGHT DOUBLE QUOTATION MARK
    0x2018: "'",  # LEFT SINGLE QUOTATION MARK
    0x2019: "'",  # RIGHT SINGLE QUOTATION MARK
}


def _strip_thinking(cleaned: str) -> str:
    """Drop a leading MedGemma reasoning block, keeping the answer after it.

    Returns the input unchanged when there is no reasoning block, and when the
    block was never closed — a reasoning block truncated by the token limit
    contains no answer to recover, so the caller should fail on the original
    text rather than on a confusingly empty string.
    """
    if _THINKING_START_RE.match(cleaned) is None:
        return cleaned
    # The opening token matches the same pattern, so search past it.
    end = _THINKING_END_RE.search(cleaned, pos=1)
    if end is None:
        return cleaned
    remainder = cleaned[end.end() :].strip()
    return remainder or cleaned


def _repair_candidates(cleaned: str) -> list[str]:
    """Progressively more aggressive repairs, cheapest first.

    A 4-bit 4B model stopped at the token limit routinely emits JSON that is
    well-formed right up to the truncation point. Rather than failing the whole
    attempt, try to close it.
    """
    candidates: list[str] = [cleaned]

    normalised = cleaned.translate(_SMART_QUOTES)
    if normalised != cleaned:
        candidates.append(normalised)

    match = _OBJECT_RE.search(normalised)
    if match is not None:
        candidates.append(match.group(0))

    # Trailing commas before a closing brace/bracket.
    for base in list(candidates):
        stripped = _TRAILING_COMMA_RE.sub(r"\1", base)
        if stripped != base:
            candidates.append(stripped)

    # Truncation: take everything from the first brace and balance it.
    start = normalised.find("{")
    if start != -1:
        repaired = _close_truncated_json(normalised[start:])
        if repaired is not None:
            candidates.append(repaired)
            candidates.append(_TRAILING_COMMA_RE.sub(r"\1", repaired))

    return candidates


def _close_truncated_json(text: str) -> str | None:
    """Balance an object that ran out of tokens mid-structure.

    Walks the text tracking string state so braces inside string literals are
    not counted, drops any dangling partial key/value, then appends the closers
    still outstanding. Returns None when nothing is outstanding.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    last_safe = 0

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack:
                stack.pop()
            if not stack:
                last_safe = index + 1
        elif char == "," and not stack[1:]:
            last_safe = index

    if not stack and last_safe:
        return None

    # Cut back to the last complete member so a half-written value is dropped.
    body = text[:last_safe].rstrip().rstrip(",") if last_safe else None
    if not body:
        return None
    return body + "".join(reversed(_closers_for(body)))


def _closers_for(body: str) -> list[str]:
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in body:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack:
            stack.pop()
    return stack

_CANONICALISE_CATEGORY: Final[dict[str, str]] = {
    "ptx": "pneumothorax",
    "pneumothraces": "pneumothorax",
    "effusion": "pleural_effusion",
    "pleural effusion": "pleural_effusion",
    "pleural-effusion": "pleural_effusion",
    "pleural_effusion": "pleural_effusion",
    "cardio": "cardiomegaly",
    "cardiomagaly": "cardiomegaly",
    "consolidations": "consolidation",
    "airspace opacity": "consolidation",
    "airspace_opacity": "consolidation",
    "nodule/mass": "nodule_or_mass",
    "nodule or mass": "nodule_or_mass",
    "nodule": "nodule_or_mass",
    "mass": "nodule_or_mass",
    "fractures": "fracture",
    "no findings": "no_finding",
    "no-finding": "no_finding",
    "no_finding": "no_finding",
    "normal": "no_finding",
}


def _format_primary_for_comparison(category: str, confidence: float, rationale: str) -> str:
    pct = round(float(confidence) * 100)
    snippet = rationale.strip()
    max_snippet = 4000 - 200
    if len(snippet) > max_snippet:
        snippet = snippet[:max_snippet] + "..."
    return f"Model primary: {category.replace('_', ' ')} (confidence {pct}%). Rationale: {snippet}"


def _parse_differentials(
    differential: object, *, ceiling: float = 1.0
) -> tuple[DifferentialFinding, ...]:
    """Parse the alternatives the model offered, bounded by `ceiling`.

    `ceiling` is the primary finding's confidence. A differential is how
    strongly a student would be drawn to a *wrong* answer, so one scoring above
    the primary is incoherent — the student would be more tempted by the error
    than by the finding the model says is demonstrated. Three prompt versions
    tried to state that invariant in words and moved the violation rate 3/6 ->
    2/6 -> 4/6 across the same six cases, with the failures clustering wherever
    the primary confidence was low. An invariant that has to hold every time
    belongs in code, so it is enforced here rather than asked for.

    Clamping rather than rejecting: the label and the ordering are the useful
    parts, and a student is better served by a differential pinned to the
    primary's confidence than by the section vanishing.
    """
    if not isinstance(differential, list) or not differential:
        return ()
    limit = min(1.0, max(0.0, float(ceiling)))
    parsed: list[DifferentialFinding] = []
    for item in differential[:3]:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        conf = item.get("confidence")
        if not isinstance(conf, int | float):
            conf = 0.0
        # A model that emits 85 instead of 0.85 should not sink the whole parse.
        clamped = min(limit, max(0.0, float(conf)))
        parsed.append(
            DifferentialFinding(label=label.strip()[:200], confidence=Confidence(clamped))
        )
    return tuple(parsed)


def _format_missed_clues(differentials: tuple[DifferentialFinding, ...]) -> str:
    if not differentials:
        return "No alternative diagnoses offered."
    parts = [f"{d.label} ({d.confidence.value * 100:.0f}%)" for d in differentials]
    return "Differentials considered: " + ", ".join(parts)
