from __future__ import annotations

import pytest

from cxt.domain.entities.finding import FindingCategory
from cxt.infrastructure.data.nih import (
    NIH_FOURTEEN,
    NO_FINDING_LABEL,
    NihLabel,
    NihMapping,
    find_primary_category,
    is_positive_for,
    nih_labels_to_finding_categories,
)


class TestNihLabel:
    def test_accepts_simple_label(self) -> None:
        assert NihLabel("Pneumothorax").value == "Pneumothorax"

    def test_strips_whitespace(self) -> None:
        assert NihLabel("  Effusion  ").value == "Effusion"

    @pytest.mark.parametrize("bad", ["", "   ", None, 123])
    def test_rejects_empty_or_non_string(self, bad) -> None:
        with pytest.raises(ValueError):
            NihLabel(bad)  # type: ignore[arg-type]

    def test_equality_with_string(self) -> None:
        assert NihLabel("Nodule") == "Nodule"

    def test_hashable(self) -> None:
        assert {NihLabel("Mass"), NihLabel("Mass")} == {NihLabel("Mass")}


class TestNihMapping:
    def test_all_six_target_findings_are_mapped(self) -> None:
        mapped = set(NihMapping.RULES.keys())
        expected = {
            FindingCategory.PNEUMOTHORAX,
            FindingCategory.PLEURAL_EFFUSION,
            FindingCategory.CARDIOMEGALY,
            FindingCategory.CONSOLIDATION,
            FindingCategory.NODULE_OR_MASS,
            FindingCategory.FRACTURE,
        }
        assert mapped == expected

    def test_consolidation_collapses_three_nih_labels(self) -> None:
        assert (
            NihLabel("Consolidation"),
            NihLabel("Infiltration"),
            NihLabel("Pneumonia"),
        ) == NihMapping.CONSOLIDATION

    def test_nodule_or_mass_collapses_two(self) -> None:
        assert (
            NihLabel("Nodule"),
            NihLabel("Mass"),
        ) == NihMapping.NODULE_OR_MASS


class TestNihLabelsToFindingCategories:
    def test_no_finding_string_maps_to_no_finding_category(self) -> None:
        assert nih_labels_to_finding_categories(["No Finding"]) == (FindingCategory.NO_FINDING,)

    def test_single_label_maps_to_one_category(self) -> None:
        assert nih_labels_to_finding_categories(["Pneumothorax"]) == (
            FindingCategory.PNEUMOTHORAX,
        )

    def test_pipeline_notation_is_split(self) -> None:
        result = nih_labels_to_finding_categories(["Cardiomegaly|Effusion"])
        assert FindingCategory.CARDIOMEGALY in result
        assert FindingCategory.PLEURAL_EFFUSION in result
        assert len(result) == 2

    def test_infiltration_and_pneumonia_collapse_into_consolidation(self) -> None:
        result = nih_labels_to_finding_categories(["Infiltration"])
        assert result == (FindingCategory.CONSOLIDATION,)

    def test_unknown_label_maps_to_no_finding(self) -> None:
        assert nih_labels_to_finding_categories(["Hernia"]) == (FindingCategory.NO_FINDING,)

    def test_mixed_known_and_unknown_only_returns_known(self) -> None:
        result = nih_labels_to_finding_categories(["Fracture", "Hernia"])
        assert result == (FindingCategory.FRACTURE,)

    def test_nodule_and_mass_both_map_to_nodule_or_mass(self) -> None:
        a = nih_labels_to_finding_categories(["Nodule"])
        b = nih_labels_to_finding_categories(["Mass"])
        assert a == b == (FindingCategory.NODULE_OR_MASS,)


class TestIsPositiveFor:
    @pytest.mark.parametrize(
        "labels,target",
        [
            (["Pneumothorax"], FindingCategory.PNEUMOTHORAX),
            (["Effusion"], FindingCategory.PLEURAL_EFFUSION),
            (["Cardiomegaly"], FindingCategory.CARDIOMEGALY),
            (["Consolidation"], FindingCategory.CONSOLIDATION),
            (["Infiltration"], FindingCategory.CONSOLIDATION),
            (["Pneumonia"], FindingCategory.CONSOLIDATION),
            (["Nodule"], FindingCategory.NODULE_OR_MASS),
            (["Mass"], FindingCategory.NODULE_OR_MASS),
            (["Fracture"], FindingCategory.FRACTURE),
        ],
    )
    def test_each_target_has_a_positive_label(self, labels, target) -> None:
        assert is_positive_for(labels, target) is True

    def test_no_finding_is_negative_for_all_targets(self) -> None:
        for cat in (
            FindingCategory.PNEUMOTHORAX,
            FindingCategory.PLEURAL_EFFUSION,
            FindingCategory.CARDIOMEGALY,
            FindingCategory.CONSOLIDATION,
            FindingCategory.NODULE_OR_MASS,
            FindingCategory.FRACTURE,
        ):
            assert is_positive_for(["No Finding"], cat) is False


class TestFindPrimaryCategory:
    def test_no_finding_when_only_no_finding(self) -> None:
        assert find_primary_category(["No Finding"]) == FindingCategory.NO_FINDING

    def test_priority_ordering(self) -> None:
        assert find_primary_category(["Cardiomegaly", "Pneumothorax"]) == FindingCategory.PNEUMOTHORAX
        assert find_primary_category(["Nodule", "Effusion"]) == FindingCategory.PLEURAL_EFFUSION
        assert find_primary_category(["Mass", "Fracture"]) == FindingCategory.FRACTURE

    def test_returns_first_mapped_category(self) -> None:
        assert find_primary_category(["Fracture"]) == FindingCategory.FRACTURE


def test_no_finding_label_constant() -> None:
    assert NO_FINDING_LABEL == "No Finding"
    assert "No Finding" not in NIH_FOURTEEN
