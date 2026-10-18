"""Compare a student's answer with the case record, finding by finding.

This is deliberately **not** a model call. Whether the student named the
confirmed finding is a question the case record already answers, and Phase 1
measured MedGemma's own recognition at 16.7-23.3% against a 14.3% chance
floor. A model deciding whether the student was right would be close to a
coin-flip, presented to the student as a verdict.

What it can and cannot do follows from `find_mentions`, which it is built on:
it matches finding names by keyword, so a misspelt finding is not recognised,
and "effusion versus consolidation" reads as naming both. It also cannot check
laterality or severity, because the NIH labels carry neither. The frontend
says so next to the result.

The comparison is computed when an attempt is read, not stored: it is a pure
function of the saved answer and the case, so older attempts get it too.
`COMPARISON_VERSION` is returned with every result so a screenshot or a
review can say which rules produced it; bump it whenever the rules or the
alias list change.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from cxt.domain.entities.case import Case
from cxt.domain.entities.finding import FindingCategory
from cxt.infrastructure.retrieval.misconception import Mention, find_mentions

__all__ = [
    "COMPARISON_VERSION",
    "AnswerComparison",
    "ComparisonItem",
    "Verdict",
    "compare_answer",
]

COMPARISON_VERSION: Final[str] = "ac1"

_LABELS: Final[dict[str, str]] = {
    FindingCategory.PNEUMOTHORAX.value: "Pneumothorax",
    FindingCategory.PLEURAL_EFFUSION.value: "Pleural effusion",
    FindingCategory.CARDIOMEGALY.value: "Cardiomegaly",
    FindingCategory.CONSOLIDATION.value: "Consolidation",
    FindingCategory.NODULE_OR_MASS.value: "Nodule or mass",
    FindingCategory.FRACTURE.value: "Fracture",
    FindingCategory.NO_FINDING.value: "No finding",
}

_NO_FINDING: Final[str] = FindingCategory.NO_FINDING.value


class Verdict(str, Enum):
    #: Named the primary confirmed finding.
    CORRECT = "correct"
    #: Named one of the case's other confirmed findings.
    ALSO_PRESENT = "also_present"
    #: Named a finding the case record does not confirm. Not "wrong": the NIH
    #: labels are mined from reports and are noisy, so the true statement is
    #: that the record does not list it.
    NOT_IN_RECORD = "not_in_record"
    #: A confirmed finding the answer never mentions.
    MISSED = "missed"
    #: A confirmed finding the answer only mentions to rule out.
    WRONGLY_EXCLUDED = "wrongly_excluded"
    #: A finding the answer rules out, and the record agrees. Credited because
    #: excluding a plausible differential is the reasoning being taught.
    CORRECTLY_EXCLUDED = "correctly_excluded"


@dataclass(frozen=True, slots=True)
class ComparisonItem:
    category: str
    label: str
    verdict: Verdict
    #: (start, end) offsets into the stored answer. Empty for MISSED.
    spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class AnswerComparison:
    items: tuple[ComparisonItem, ...]
    #: False when no finding name was recognised at all. The frontend then
    #: says so instead of listing every confirmed finding as missed, because a
    #: student who wrote "pneumothrax" did not miss the pneumothorax.
    matched_anything: bool
    version: str = COMPARISON_VERSION


def _spans(mentions: list[Mention]) -> tuple[tuple[int, int], ...]:
    return tuple((m.start, m.end) for m in mentions)


def _is_asserted(mention: Mention) -> bool:
    # "No abnormality" and "no acute abnormality" are no-finding aliases that
    # begin with the negation cue "no ", so the negation pass always marks them
    # negated. For this one category that is the alias negating itself, not
    # the student ruling a normal film out, so it counts as asserted.
    return mention.category == _NO_FINDING or not mention.negated


def compare_answer(free_text: str, case: Case) -> AnswerComparison:
    mentions = find_mentions(free_text)
    if not mentions:
        return AnswerComparison(items=(), matched_anything=False)

    asserted: dict[str, list[Mention]] = {}
    excluded: dict[str, list[Mention]] = {}
    for mention in mentions:
        bucket = asserted if _is_asserted(mention) else excluded
        bucket.setdefault(mention.category, []).append(mention)
    # Positive mentions win: "no effusion on the left; effusion on the right"
    # names an effusion.
    for category in asserted:
        excluded.pop(category, None)

    primary = case.ground_truth_finding.category.value
    labels = {primary: case.ground_truth_finding.label.value}
    for extra in case.secondary_findings:
        labels[extra.category.value] = extra.label.value

    items: list[ComparisonItem] = []

    def add(category: str, verdict: Verdict, found: list[Mention]) -> None:
        label = labels.get(category) or _LABELS.get(category, category)
        items.append(ComparisonItem(category, label, verdict, _spans(found)))

    confirmed = case.confirmed_categories
    for category in confirmed:
        if category in asserted:
            verdict = Verdict.CORRECT if category == primary else Verdict.ALSO_PRESENT
            add(category, verdict, asserted[category])
        elif category == _NO_FINDING and not any(c != _NO_FINDING for c in asserted):
            # A normal film described only by what is absent ("no effusion, no
            # pneumothorax, heart size normal") is a correct normal read even
            # though it never says "normal".
            add(category, Verdict.CORRECT, [])
        elif category in excluded:
            add(category, Verdict.WRONGLY_EXCLUDED, excluded[category])
        else:
            add(category, Verdict.MISSED, [])

    for category, found in asserted.items():
        if category not in confirmed:
            add(category, Verdict.NOT_IN_RECORD, found)
    for category, found in excluded.items():
        if category not in confirmed:
            add(category, Verdict.CORRECTLY_EXCLUDED, found)

    return AnswerComparison(items=tuple(items), matched_anything=True)
