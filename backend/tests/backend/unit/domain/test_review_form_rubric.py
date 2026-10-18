"""The educator review form must send the rubric the domain accepts.

There are no frontend tests in this project, so nothing but this file stands
between the two lists and a silent divergence. It diverged once already: the
form submitted `image_quality_assessment`, `anatomical_observation`,
`pathology_identification`, `differential_ranking`, `clinical_reasoning` and
`report_clarity`, none of which `_VALID_CRITERIA` allows, so every review
submitted through the UI was rejected with HTTP 422 and FR-7 was unreachable
by the only route a real educator has.

Reading a `.tsx` file from a Python test is unusual and deliberate: the
alternative is serving the rubric from an endpoint the form fetches, which is
more machinery than one hard-coded list of six strings justifies. This test is
the cheap version of the same guarantee.
"""
from __future__ import annotations

import re
from pathlib import Path

from cxt.domain.entities.review import _VALID_CRITERIA

# tests/backend/unit/domain/ -> backend/ -> repository root.
REVIEW_PAGE = (
    Path(__file__).resolve().parents[5] / "frontend" / "src" / "pages" / "ReviewPage.tsx"
)

_RUBRIC_BLOCK = re.compile(r"const RUBRIC = \[(.*?)\] as const;", re.DOTALL)


def _form_criteria() -> set[str]:
    # Deliberately not skipped when the file is missing: a skip here would
    # report green while the guard did nothing, which is how the drift this
    # test exists to catch would get through a second time.
    assert REVIEW_PAGE.exists(), f"Expected the review form at {REVIEW_PAGE}"
    match = _RUBRIC_BLOCK.search(REVIEW_PAGE.read_text(encoding="utf-8"))
    assert match is not None, "ReviewPage.tsx no longer declares `const RUBRIC = [...]`"
    return set(re.findall(r'"([a-z_]+)"', match.group(1)))


def test_the_review_form_sends_exactly_the_criteria_the_domain_accepts() -> None:
    assert _form_criteria() == set(_VALID_CRITERIA)
