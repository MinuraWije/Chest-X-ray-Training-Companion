"""One test per comparison rule.

The verdicts are shown to a student as a judgement of their answer, so each
rule is pinned here with the sentence that motivates it.
"""
from __future__ import annotations

from cxt.application.services.answer_comparison import (
    COMPARISON_VERSION,
    Verdict,
    compare_answer,
)
from cxt.domain.entities.case import Case
from cxt.domain.entities.finding import Finding, FindingCategory
from cxt.domain.value_objects.finding_label import FindingLabel


def _case(primary: FindingCategory, *secondary: FindingCategory) -> Case:
    def finding(category: FindingCategory) -> Finding:
        return Finding(category=category, label=FindingLabel(category.value.replace("_", " ")))

    return Case(
        id="case_test",
        image_url="/images/x.png",
        ground_truth_finding=finding(primary),
        secondary_findings=tuple(finding(c) for c in secondary),
    )


def _verdicts(text: str, case: Case) -> dict[str, Verdict]:
    return {item.category: item.verdict for item in compare_answer(text, case).items}


class TestConfirmedFindings:
    def test_naming_the_primary_is_correct(self) -> None:
        text = "There is a right apical pneumothorax."
        result = compare_answer(text, _case(FindingCategory.PNEUMOTHORAX))

        (item,) = result.items
        assert item.verdict is Verdict.CORRECT
        assert [text[s:e] for s, e in item.spans] == ["pneumothorax"]
        assert result.version == COMPARISON_VERSION

    def test_the_label_comes_from_the_case_record(self) -> None:
        (item,) = compare_answer(
            "Pneumothorax on the right.", _case(FindingCategory.PNEUMOTHORAX)
        ).items

        assert item.label == "pneumothorax"

    def test_naming_a_secondary_finding_is_also_present(self) -> None:
        case = _case(FindingCategory.PNEUMOTHORAX, FindingCategory.PLEURAL_EFFUSION)

        verdicts = _verdicts("A pneumothorax with a small pleural effusion.", case)

        assert verdicts == {
            "pneumothorax": Verdict.CORRECT,
            "pleural_effusion": Verdict.ALSO_PRESENT,
        }

    def test_a_confirmed_finding_never_mentioned_is_missed(self) -> None:
        case = _case(FindingCategory.PNEUMOTHORAX, FindingCategory.PLEURAL_EFFUSION)
        result = compare_answer("There is a pneumothorax.", case)

        missed = [i for i in result.items if i.verdict is Verdict.MISSED]
        assert [i.category for i in missed] == ["pleural_effusion"]
        assert missed[0].spans == ()

    def test_ruling_out_a_confirmed_finding_is_wrongly_excluded(self) -> None:
        verdicts = _verdicts(
            "There is no pneumothorax. The heart is enlarged.",
            _case(FindingCategory.PNEUMOTHORAX),
        )

        assert verdicts["pneumothorax"] is Verdict.WRONGLY_EXCLUDED
        assert verdicts["cardiomegaly"] is Verdict.NOT_IN_RECORD

    def test_a_positive_mention_outweighs_a_negated_one(self) -> None:
        verdicts = _verdicts(
            "No effusion on the left; small effusion on the right.",
            _case(FindingCategory.PLEURAL_EFFUSION),
        )

        assert verdicts == {"pleural_effusion": Verdict.CORRECT}


class TestOtherFindings:
    def test_naming_an_unconfirmed_finding_is_not_in_record(self) -> None:
        verdicts = _verdicts(
            "Cardiomegaly and a pneumothorax.", _case(FindingCategory.PNEUMOTHORAX)
        )

        assert verdicts["cardiomegaly"] is Verdict.NOT_IN_RECORD

    def test_ruling_out_an_unconfirmed_finding_is_credited(self) -> None:
        verdicts = _verdicts(
            "Pneumothorax at the apex. There is no pleural effusion.",
            _case(FindingCategory.PNEUMOTHORAX),
        )

        assert verdicts["pleural_effusion"] is Verdict.CORRECTLY_EXCLUDED

    def test_items_list_confirmed_findings_first(self) -> None:
        case = _case(FindingCategory.PNEUMOTHORAX, FindingCategory.PLEURAL_EFFUSION)
        items = compare_answer(
            "Cardiomegaly. No nodule. Effusion and pneumothorax.", case
        ).items

        assert [i.category for i in items] == [
            "pneumothorax",
            "pleural_effusion",
            "cardiomegaly",
            "nodule_or_mass",
        ]


class TestNormalFilms:
    def test_calling_a_normal_film_normal_is_correct(self) -> None:
        verdicts = _verdicts("Unremarkable chest film.", _case(FindingCategory.NO_FINDING))

        assert verdicts == {"no_finding": Verdict.CORRECT}

    def test_no_abnormality_counts_as_calling_it_normal(self) -> None:
        # The alias begins with the negation cue "no "; it must not be read as
        # the student ruling a normal film out.
        verdicts = _verdicts(
            "No acute abnormality on this radiograph.", _case(FindingCategory.NO_FINDING)
        )

        assert verdicts == {"no_finding": Verdict.CORRECT}

    def test_ruling_everything_out_is_a_correct_normal_read(self) -> None:
        verdicts = _verdicts(
            "There is no pneumothorax. There is no pleural effusion.",
            _case(FindingCategory.NO_FINDING),
        )

        assert verdicts == {
            "no_finding": Verdict.CORRECT,
            "pneumothorax": Verdict.CORRECTLY_EXCLUDED,
            "pleural_effusion": Verdict.CORRECTLY_EXCLUDED,
        }

    def test_reporting_an_abnormality_on_a_normal_film(self) -> None:
        verdicts = _verdicts(
            "There is a large right pleural effusion.", _case(FindingCategory.NO_FINDING)
        )

        assert verdicts == {
            "no_finding": Verdict.MISSED,
            "pleural_effusion": Verdict.NOT_IN_RECORD,
        }

    def test_calling_an_abnormal_film_normal(self) -> None:
        verdicts = _verdicts(
            "The lungs are clear. Unremarkable film.", _case(FindingCategory.PNEUMOTHORAX)
        )

        assert verdicts == {
            "pneumothorax": Verdict.MISSED,
            "no_finding": Verdict.NOT_IN_RECORD,
        }


class TestNothingRecognised:
    def test_an_unrecognised_answer_reports_nothing_missed(self) -> None:
        # A misspelling is not recognised. Telling this student they missed the
        # pneumothorax would contradict what they actually wrote.
        result = compare_answer(
            "Left apical pneumothrax with a visible edge.", _case(FindingCategory.PNEUMOTHORAX)
        )

        assert result.matched_anything is False
        assert result.items == ()
