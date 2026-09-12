"""Lexical detection of which findings a student named — and which they ruled out.

This is a **lexical proxy for** misconception detection, and the write-up must
call it that. Prose-to-misconception is itself a classification problem, and
this model measured 16-25% on an easier seven-way task; asking it to label the
student's error would put a coin-flip inside the retrieval signal and make
every downstream number un-attributable. Forty auditable lines of alias
matching is a weaker instrument that can at least be inspected, and its failure
cases can be listed.

**Negation is the nastiest bug here, not an edge case.** "There is no
pneumothorax" contains the word `pneumothorax`. Matched naively, a student who
correctly ruled out a pneumothorax is recorded as having claimed one, and the
misconception arm then retrieves pneumothorax material to correct an error they
did not make — the system confidently teaching against a correct answer. So
negation is detected first and the span is removed before alias matching runs.

Known failure cases, stated rather than hidden:
- "Effusion versus consolidation" is read as naming both; a student weighing a
  differential looks the same as one asserting two findings.
- Negation is scoped by a fixed window, so "no focal consolidation, but there
  is an effusion" is handled, while a negation split across a long clause is
  not.
- Misspellings do not match. A student who writes "pneumothrax" is recorded as
  naming nothing.
"""
from __future__ import annotations

import re
from typing import Final

from cxt.domain.entities.finding import FindingCategory

__all__ = [
    "CATEGORY_ALIASES",
    "named_categories",
    "negated_categories",
    "strip_negated_spans",
    "mentions_category",
]

#: Surface forms a student plausibly writes for each category. Matched
#: case-insensitively on word boundaries. Deliberately not stemmed: "consolidated"
#: and "consolidation" are both listed rather than inferred, so the list is
#: auditable and a reader can see exactly what does and does not match.
CATEGORY_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    FindingCategory.PNEUMOTHORAX.value: (
        "pneumothorax",
        "pneumothoraces",
        "collapsed lung",
        "lung collapse",
        "visceral pleural line",
        "deep sulcus",
    ),
    FindingCategory.PLEURAL_EFFUSION.value: (
        "pleural effusion",
        "effusion",
        "effusions",
        "pleural fluid",
        "fluid in the pleural space",
        "blunted costophrenic",
        "blunting of the costophrenic",
        "meniscus",
        "haemothorax",
        "hemothorax",
    ),
    FindingCategory.CARDIOMEGALY.value: (
        "cardiomegaly",
        "enlarged heart",
        "heart is enlarged",
        "enlarged cardiac silhouette",
        "increased cardiothoracic ratio",
        "cardiothoracic ratio",
        "big heart",
    ),
    FindingCategory.CONSOLIDATION.value: (
        "consolidation",
        "consolidated",
        "air bronchogram",
        "airspace opacity",
        "air space opacity",
        "pneumonia",
        "infective change",
    ),
    FindingCategory.NODULE_OR_MASS.value: (
        "nodule",
        "nodules",
        "nodular",
        "mass",
        "masses",
        "lesion",
        "coin lesion",
        "tumour",
        "tumor",
        "malignancy",
    ),
    FindingCategory.FRACTURE.value: (
        "fracture",
        "fractured",
        "broken rib",
        "rib fracture",
    ),
    FindingCategory.NO_FINDING.value: (
        "no abnormality",
        "no acute abnormality",
        "normal chest",
        "normal film",
        "unremarkable",
        "clear lung fields",
        "lungs are clear",
    ),
}

#: Cues that negate what follows them. Ordered longest-first at match time so
#: "no evidence of" wins over "no".
_NEGATION_CUES: Final[tuple[str, ...]] = (
    "no evidence of",
    "no definite",
    "no obvious",
    "no significant",
    "without any",
    "not consistent with",
    "cannot see any",
    "cannot see a",
    "i do not see",
    "i don't see",
    "there is no",
    "there are no",
    "rather than",
    "absence of",
    "free of",
    "without",
    "denies",
    "unlikely",
    "no ",
    "not ",
)

#: Cues that negate what came BEFORE them. English puts negation on either
#: side of the thing negated — "no pneumothorax" and "pneumothorax is ruled
#: out" mean the same — and a forward-only scanner silently gets the second
#: form backwards, recording the student as claiming exactly what they
#: excluded. These are matched looking left.
_POSTPOSED_CUES: Final[tuple[str, ...]] = (
    "is ruled out",
    "was ruled out",
    "are ruled out",
    "is excluded",
    "can be excluded",
    "is absent",
    "are absent",
    "is not present",
    "are not present",
    "is not seen",
    "is unlikely",
    "seems unlikely",
    "ruled out",
    "excluded",
)

#: How far after a negation cue the negation reaches, in characters. A clause,
#: not a sentence: "no pneumothorax, but there is a large effusion" must negate
#: only the first finding. Tuned to roughly one clause of clinical prose.
NEGATION_WINDOW: Final[int] = 40

#: Cues that end a negation early, so the window never swallows a contrast.
_NEGATION_TERMINATORS: Final[re.Pattern[str]] = re.compile(
    r"\b(but|however|although|though|whereas|instead)\b|[.;]"
)

def _alternation(cues: tuple[str, ...]) -> re.Pattern[str]:
    """Longest-first, so "no evidence of" wins over the "no " prefix."""
    return re.compile(
        "|".join(re.escape(cue) for cue in sorted(cues, key=len, reverse=True)),
        re.IGNORECASE,
    )


_CUE_RE: Final[re.Pattern[str]] = _alternation(_NEGATION_CUES)
_POSTPOSED_RE: Final[re.Pattern[str]] = _alternation(_POSTPOSED_CUES)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def strip_negated_spans(text: str) -> str:
    """Blank out the clause following each negation cue.

    Replaced with spaces rather than deleted so any offsets a caller holds stay
    valid, and so two adjacent clauses cannot be fused into a phrase that was
    never written.
    """
    normalised = _normalise(text)
    chars = list(normalised)

    def _blank(start: int, end: int) -> None:
        for i in range(max(start, 0), min(end, len(chars))):
            chars[i] = " "

    # Forward: "no pneumothorax", "no evidence of an effusion".
    for match in _CUE_RE.finditer(normalised):
        window = normalised[match.end() : match.end() + NEGATION_WINDOW]
        terminator = _NEGATION_TERMINATORS.search(window)
        end = match.end() + (terminator.start() if terminator else len(window))
        _blank(match.start(), end)

    # Backward: "pneumothorax is ruled out". Scanned on the original text, not
    # the partly-blanked copy, so a sentence carrying both forms still works.
    for match in _POSTPOSED_RE.finditer(normalised):
        window_start = max(match.start() - NEGATION_WINDOW, 0)
        window = normalised[window_start : match.start()]
        # The nearest clause boundary to the left bounds the negation, so
        # "there is consolidation; pneumothorax is excluded" keeps the first.
        boundaries = list(_NEGATION_TERMINATORS.finditer(window))
        start = window_start + (boundaries[-1].end() if boundaries else 0)
        _blank(start, match.end())

    return "".join(chars)


def mentions_category(text: str, category: str) -> bool:
    """True when `text` contains any alias for `category`, negation ignored."""
    haystack = _normalise(text)
    for alias in CATEGORY_ALIASES.get(category, ()):
        if re.search(rf"\b{re.escape(alias)}\b", haystack):
            return True
    return False


def named_categories(text: str) -> tuple[str, ...]:
    """Categories the student asserted, with negated mentions excluded.

    Order follows `CATEGORY_ALIASES` rather than position in the text, so the
    result is stable regardless of how the student phrased their answer.
    """
    if not text or not text.strip():
        return ()
    positive = strip_negated_spans(text)
    return tuple(
        category
        for category in CATEGORY_ALIASES
        if mentions_category(positive, category)
    )


def negated_categories(text: str) -> tuple[str, ...]:
    """Categories the student explicitly ruled out.

    Worth surfacing separately: a student who correctly excludes a plausible
    differential is demonstrating exactly the reasoning the tool is meant to
    teach, and feedback that ignores it reads as if they said nothing.
    """
    if not text or not text.strip():
        return ()
    asserted = set(named_categories(text))
    return tuple(
        category
        for category in CATEGORY_ALIASES
        if category not in asserted and mentions_category(text, category)
    )
