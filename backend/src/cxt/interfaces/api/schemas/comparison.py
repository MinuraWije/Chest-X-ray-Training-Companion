"""The answer comparison -> API shape, shared by the student and educator routes.

One function for both, for the reason `shape_feedback` is one function: an
educator marking an attempt must see the same comparison the student saw.
"""
from __future__ import annotations

from cxt.application.services.answer_comparison import AnswerComparison, compare_answer
from cxt.domain.entities.student_attempt import StudentAttempt
from cxt.domain.ports.case_repository import ICaseRepository
from cxt.shared.errors import NotFoundError

__all__ = ["comparison_for", "shape_comparison"]


def shape_comparison(comparison: AnswerComparison) -> dict[str, object]:
    return {
        "version": comparison.version,
        "matchedAnything": comparison.matched_anything,
        "items": [
            {
                "category": item.category,
                "label": item.label,
                "verdict": item.verdict.value,
                "spans": [{"start": start, "end": end} for start, end in item.spans],
            }
            for item in comparison.items
        ],
    }


def comparison_for(
    attempt: StudentAttempt, cases: ICaseRepository | None
) -> dict[str, object] | None:
    """The shaped comparison, or None when the case can no longer be found.

    An attempt outlives its manifest row (the curated subset can be rebuilt),
    and a missing case must leave the attempt readable rather than fail it:
    the frontend falls back to showing the answer without highlights.
    """
    if cases is None:
        return None
    try:
        case = cases.get(attempt.case_id)
    except NotFoundError:
        return None
    return shape_comparison(compare_answer(attempt.free_text, case))
