"""Proxy-versus-human agreement, and the kappa that keeps it honest.

The important test here is `test_kappa_is_zero_when_the_proxy_says_relevant_to_everything`.
Raw agreement on this corpus is inflated by design — most retrieved chunks on a
five-finding corpus are topically related — so a proxy with no discriminating
power still scores well. If kappa did not catch that, the report would quote a
high agreement figure for a metric that answers "yes" unconditionally.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cxt.infrastructure.eval.relevance_labels import (
    MIN_PAIRS_FOR_AGREEMENT,
    VALID_LABELS,
    LabelledPair,
    export_labelling_sheet,
    load_labels,
    render_agreement_summary,
    validate_proxy,
)
from cxt.shared.errors import ValidationError

REAL_SHEET = (
    Path(__file__).resolve().parents[5].parent / "data" / "eval" / "relevance_labels.json"
)


def _pair(
    *,
    proxy: bool,
    label: str | None,
    chunk_id: str = "chunk-1",
    aspect: str = "radiographic_sign",
    case_id: str = "case-1",
    archetype: str = "correct",
) -> LabelledPair:
    return LabelledPair(
        case_id=case_id,
        category="pleural_effusion",
        archetype=archetype,
        chunk_id=chunk_id,
        chunk_aspect=aspect,
        rank=1,
        proxy_relevant=proxy,
        label=label,
    )


class TestLabelValidation:
    def test_valid_labels(self) -> None:
        assert {"relevant", "not_relevant", "unsure"} == VALID_LABELS

    def test_an_invalid_label_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="is not one of"):
            _pair(proxy=True, label="maybe")

    def test_unlabelled_is_allowed(self) -> None:
        assert _pair(proxy=True, label=None).is_labelled is False


class TestAgreement:
    def test_perfect_agreement(self) -> None:
        pairs = (
            _pair(proxy=True, label="relevant"),
            _pair(proxy=False, label="not_relevant", chunk_id="c2"),
        )
        result = validate_proxy(pairs)
        assert result.agreement == 1.0
        assert result.cohens_kappa == pytest.approx(1.0)

    def test_total_disagreement(self) -> None:
        pairs = (
            _pair(proxy=True, label="not_relevant"),
            _pair(proxy=False, label="relevant", chunk_id="c2"),
        )
        assert validate_proxy(pairs).agreement == 0.0

    def test_unsure_is_excluded_not_counted_against(self) -> None:
        """Forcing a binary on an ambiguous pair manufactures a verdict the
        labeller does not hold."""
        pairs = (
            _pair(proxy=True, label="relevant"),
            _pair(proxy=True, label="unsure", chunk_id="c2"),
        )
        result = validate_proxy(pairs)
        assert result.labelled == 2
        assert result.unsure == 1
        assert result.decisive == 1
        assert result.agreement == 1.0

    def test_unlabelled_pairs_are_not_counted(self) -> None:
        pairs = (
            _pair(proxy=True, label="relevant"),
            _pair(proxy=True, label=None, chunk_id="c2"),
        )
        result = validate_proxy(pairs)
        assert result.total_pairs == 2
        assert result.labelled == 1

    def test_confusion_cells_are_populated(self) -> None:
        pairs = (
            _pair(proxy=True, label="relevant"),
            _pair(proxy=True, label="not_relevant", chunk_id="c2"),
            _pair(proxy=False, label="relevant", chunk_id="c3"),
            _pair(proxy=False, label="not_relevant", chunk_id="c4"),
        )
        result = validate_proxy(pairs)
        assert (result.both_relevant, result.proxy_only) == (1, 1)
        assert (result.human_only, result.both_irrelevant) == (1, 1)
        assert result.agreement == pytest.approx(0.5)

    def test_agreement_is_broken_down_by_aspect(self) -> None:
        """The proxy is expected to fail differently on `management` chunks —
        topical and unhelpful — than on radiographic signs."""
        pairs = (
            _pair(proxy=True, label="relevant", aspect="radiographic_sign"),
            _pair(proxy=True, label="not_relevant", chunk_id="c2", aspect="management"),
        )
        result = validate_proxy(pairs)
        assert result.by_aspect["radiographic_sign"] == (1, 1)
        assert result.by_aspect["management"] == (0, 1)

    def test_no_labels_is_zero_not_a_crash(self) -> None:
        result = validate_proxy((_pair(proxy=True, label=None),))
        assert result.agreement == 0.0
        assert result.cohens_kappa == 0.0


class TestKappa:
    def test_kappa_is_zero_when_the_proxy_says_relevant_to_everything(self) -> None:
        """The test this file exists for.

        Nine of ten pairs are genuinely relevant, and a proxy that answers
        "relevant" unconditionally agrees on nine of them — 90% raw agreement
        for an instrument with no discriminating power whatsoever. Kappa must
        expose that; raw agreement alone would put a flattering and false
        number in the report.
        """
        pairs = tuple(
            _pair(
                proxy=True,
                label="relevant" if i < 9 else "not_relevant",
                chunk_id=f"c{i}",
            )
            for i in range(10)
        )
        result = validate_proxy(pairs)
        assert result.agreement == pytest.approx(0.9)
        assert result.cohens_kappa == pytest.approx(0.0)

    def test_kappa_below_raw_agreement_on_a_skewed_set(self) -> None:
        pairs = tuple(
            _pair(
                proxy=i < 8,
                label="relevant" if i < 9 else "not_relevant",
                chunk_id=f"c{i}",
            )
            for i in range(10)
        )
        result = validate_proxy(pairs)
        assert result.cohens_kappa < result.agreement

    def test_kappa_is_defined_for_a_balanced_perfect_rater(self) -> None:
        pairs = tuple(
            _pair(proxy=i < 5, label="relevant" if i < 5 else "not_relevant", chunk_id=f"c{i}")
            for i in range(10)
        )
        assert validate_proxy(pairs).cohens_kappa == pytest.approx(1.0)


class TestQuotability:
    def test_a_thin_sample_is_flagged_as_not_quotable(self) -> None:
        pairs = tuple(
            _pair(proxy=True, label="relevant", chunk_id=f"c{i}") for i in range(5)
        )
        result = validate_proxy(pairs)
        assert not result.is_quotable
        assert "Not quotable" in render_agreement_summary(result)

    def test_a_full_sample_is_quotable(self) -> None:
        pairs = tuple(
            _pair(proxy=True, label="relevant", chunk_id=f"c{i}")
            for i in range(MIN_PAIRS_FOR_AGREEMENT)
        )
        assert validate_proxy(pairs).is_quotable


class TestSheetRoundTrip:
    def test_export_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.json"
        export_labelling_sheet([_pair(proxy=True, label=None)], path)
        assert len(load_labels(path)) == 1

    def test_reexport_preserves_existing_labels(self, tmp_path: Path) -> None:
        """Re-running after a retrieval change must not discard the work."""
        path = tmp_path / "labels.json"
        export_labelling_sheet([_pair(proxy=True, label=None)], path)
        export_labelling_sheet([_pair(proxy=True, label="relevant")], path)
        export_labelling_sheet([_pair(proxy=False, label=None)], path)
        reloaded = load_labels(path)
        assert reloaded[0].label == "relevant"
        # The human verdict is kept; the proxy's is refreshed, since the point
        # is to see whether the proxy moved toward the human.
        assert reloaded[0].proxy_relevant is False

    def test_malformed_sheet_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"nope": []}), encoding="utf-8")
        with pytest.raises(ValidationError, match="'pairs' list"):
            load_labels(path)


class TestTheCommittedSheet:
    """The sheet in the repository must be real, unlabelled work-in-progress."""

    def test_it_exists_and_parses(self) -> None:
        assert REAL_SHEET.exists()
        assert len(load_labels(REAL_SHEET)) > 0

    def test_it_contains_no_fabricated_labels(self) -> None:
        """No label in this repository may be anything but a human's judgement.

        Phase 2 measures citation fidelity. Committing machine-generated labels
        as if a person had made them would corrupt the one number the phase
        exists to report.
        """
        for pair in load_labels(REAL_SHEET):
            assert pair.label is None, (
                f"{pair.chunk_id} carries a label. If this is genuine human "
                "work, delete this assertion deliberately."
            )

    def test_archetypes_are_balanced(self) -> None:
        """A truncated sample must span error types, not be 150 correct answers."""
        counts: dict[str, int] = {}
        for pair in load_labels(REAL_SHEET):
            counts[pair.archetype] = counts.get(pair.archetype, 0) + 1
        assert len(counts) >= 6
        assert max(counts.values()) - min(counts.values()) <= 5
