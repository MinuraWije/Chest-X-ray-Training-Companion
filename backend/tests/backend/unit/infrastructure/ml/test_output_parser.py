from __future__ import annotations

import json

import pytest

from cxt.infrastructure.ml.output_parser import OutputParser
from cxt.shared.errors import UpstreamUnavailableError


def _good_payload(category: str = "pneumothorax") -> dict[str, object]:
    return {
        "primary_finding": {
            "category": category,
            "label": "Right apical pneumothorax",
            "confidence": 0.82,
            "rationale": "Absent lung markings in the right upper zone.",
        },
        "differential_findings": [
            {"label": "Apical bulla", "confidence": 0.10},
        ],
        "key_observations": "Visceral pleural line in the apex.",
        "teaching_point": "Look for the pleural line in the apex on PA.",
    }


class TestOutputParserRepairsImperfectJson:
    """A 4-bit 4B model rarely emits a pristine object; static repair beats a 503."""

    def test_repairs_trailing_comma(self) -> None:
        raw = json.dumps(_good_payload())[:-1] + ",}"
        feedback = OutputParser().parse(attempt_id="a1", raw_text=raw)
        assert feedback.likely_finding.value == "Right apical pneumothorax"

    def test_repairs_smart_quotes(self) -> None:
        raw = json.dumps(_good_payload()).replace(
            '"Right apical pneumothorax"', "“Right apical pneumothorax”"
        )
        feedback = OutputParser().parse(attempt_id="a1", raw_text=raw)
        assert feedback.likely_finding.value == "Right apical pneumothorax"

    def test_repairs_output_truncated_at_the_token_limit(self) -> None:
        raw = json.dumps(_good_payload())[:-40]
        feedback = OutputParser().parse(attempt_id="a1", raw_text=raw)
        # The primary finding survives even though the tail was lost.
        assert feedback.likely_finding.value == "Right apical pneumothorax"
        assert feedback.confidence.value == 0.82

    def test_repairs_unterminated_code_fence(self) -> None:
        raw = "```json\n" + json.dumps(_good_payload())[:-30]
        feedback = OutputParser().parse(attempt_id="a1", raw_text=raw)
        assert feedback.likely_finding.value == "Right apical pneumothorax"

    def test_strips_conversational_prose_around_the_object(self) -> None:
        raw = "Sure! Here is my analysis:\n" + json.dumps(_good_payload()) + "\nHope this helps."
        feedback = OutputParser().parse(attempt_id="a1", raw_text=raw)
        assert feedback.likely_finding.value == "Right apical pneumothorax"

    def test_truncation_before_the_primary_finding_still_fails(self) -> None:
        # Repair must not invent a finding out of an unusable fragment.
        with pytest.raises(UpstreamUnavailableError):
            OutputParser().parse(attempt_id="a1", raw_text='{"primary_fin')


class TestOutputParserPreservesModelDetail:
    def test_key_observations_reach_the_entity(self) -> None:
        feedback = OutputParser().parse(
            attempt_id="a1", raw_text=json.dumps(_good_payload())
        )
        assert feedback.key_observations is not None
        assert feedback.key_observations.value == "Visceral pleural line in the apex."

    def test_differentials_are_structured_not_just_prose(self) -> None:
        feedback = OutputParser().parse(
            attempt_id="a1", raw_text=json.dumps(_good_payload())
        )
        assert len(feedback.differentials) == 1
        assert feedback.differentials[0].label == "Apical bulla"
        assert feedback.differentials[0].confidence.value == 0.10
        assert "Apical bulla (10%)" in feedback.missed_clues.value

    def test_out_of_range_differential_confidence_is_clamped_not_fatal(self) -> None:
        payload = _good_payload()
        payload["differential_findings"] = [{"label": "Bulla", "confidence": 85}]
        feedback = OutputParser().parse(attempt_id="a1", raw_text=json.dumps(payload))
        # 85 is read as a percent-scale slip and survives the parse; the ceiling
        # it lands on is the primary's confidence, not 1.0, since a differential
        # may not outrank the finding it is an alternative to.
        assert feedback.differentials[0].confidence.value == pytest.approx(0.82)

    def test_missing_differentials_yield_an_empty_tuple(self) -> None:
        payload = _good_payload()
        payload["differential_findings"] = []
        feedback = OutputParser().parse(attempt_id="a1", raw_text=json.dumps(payload))
        assert feedback.differentials == ()
        assert feedback.missed_clues.value == "No alternative diagnoses offered."


class TestOutputParserHappyPath:
    def test_parses_clean_json(self) -> None:
        parser = OutputParser()
        feedback = parser.parse(
            attempt_id="a1",
            raw_text=json.dumps(_good_payload()),
        )
        assert feedback.likely_finding.value == "Right apical pneumothorax"
        assert feedback.confidence.value == 0.82
        assert feedback.key_signs.value == "Absent lung markings in the right upper zone."
        assert feedback.sources == ()

    def test_strips_markdown_fences(self) -> None:
        parser = OutputParser()
        raw = "```json\n" + json.dumps(_good_payload()) + "\n```"
        feedback = parser.parse(attempt_id="a1", raw_text=raw)
        assert feedback.likely_finding.value == "Right apical pneumothorax"

    def test_tolerates_prose_around_json(self) -> None:
        parser = OutputParser()
        raw = "Here is the JSON:\n" + json.dumps(_good_payload()) + "\nHope that helps."
        feedback = parser.parse(attempt_id="a1", raw_text=raw)
        assert feedback.likely_finding.value == "Right apical pneumothorax"

    def test_canonicalises_category_aliases(self) -> None:
        parser = OutputParser()
        for alias in ("pneumothraces", "pTx", "PTX"):
            payload = _good_payload()
            assert isinstance(payload["primary_finding"], dict)
            payload["primary_finding"]["category"] = alias
            feedback = parser.parse(attempt_id="a1", raw_text=json.dumps(payload))
            assert feedback.attempt_id == "a1"

    def test_handles_all_seven_categories(self) -> None:
        parser = OutputParser()
        for cat in (
            "pneumothorax",
            "pleural_effusion",
            "cardiomegaly",
            "consolidation",
            "nodule_or_mass",
            "fracture",
            "no_finding",
        ):
            feedback = parser.parse(
                attempt_id="a1", raw_text=json.dumps(_good_payload(cat))
            )
            assert feedback.attempt_id == "a1"

    def test_truncates_overlong_rationale(self) -> None:
        parser = OutputParser()
        payload = _good_payload()
        assert isinstance(payload["primary_finding"], dict)
        payload["primary_finding"]["rationale"] = "x" * 8000
        feedback = parser.parse(attempt_id="a1", raw_text=json.dumps(payload))
        assert feedback.key_signs.value
        assert len(feedback.key_signs.value) <= 500


class TestOutputParserErrors:
    def test_empty_output_raises_upstream(self) -> None:
        parser = OutputParser()
        with pytest.raises(UpstreamUnavailableError):
            parser.parse(attempt_id="a1", raw_text="")

    def test_no_json_raises_upstream(self) -> None:
        parser = OutputParser()
        with pytest.raises(UpstreamUnavailableError):
            parser.parse(attempt_id="a1", raw_text="just plain text with no json")

    def test_invalid_json_raises_upstream(self) -> None:
        parser = OutputParser()
        with pytest.raises(UpstreamUnavailableError):
            parser.parse(attempt_id="a1", raw_text='{"primary_finding": ')

    def test_missing_primary_finding_raises(self) -> None:
        parser = OutputParser()
        raw = json.dumps({"key_observations": "x", "teaching_point": "y"})
        with pytest.raises(UpstreamUnavailableError):
            parser.parse(attempt_id="a1", raw_text=raw)

    def test_unknown_category_raises(self) -> None:
        parser = OutputParser()
        payload = _good_payload()
        assert isinstance(payload["primary_finding"], dict)
        payload["primary_finding"]["category"] = "splinter"
        with pytest.raises(UpstreamUnavailableError):
            parser.parse(attempt_id="a1", raw_text=json.dumps(payload))

    def test_non_numeric_confidence_raises(self) -> None:
        parser = OutputParser()
        payload = _good_payload()
        assert isinstance(payload["primary_finding"], dict)
        payload["primary_finding"]["confidence"] = "high"
        with pytest.raises(UpstreamUnavailableError):
            parser.parse(attempt_id="a1", raw_text=json.dumps(payload))

    def test_missing_label_falls_back_to_category_title(self) -> None:
        parser = OutputParser()
        payload = _good_payload()
        assert isinstance(payload["primary_finding"], dict)
        del payload["primary_finding"]["label"]
        feedback = parser.parse(attempt_id="a1", raw_text=json.dumps(payload))
        assert feedback.likely_finding.value == "Pneumothorax"


class TestOutputParserRawPreserved:
    def test_raw_contains_text_and_parsed(self) -> None:
        parser = OutputParser()
        text = json.dumps(_good_payload())
        feedback = parser.parse(attempt_id="a1", raw_text=text)
        assert feedback.raw["raw_text"] == text
        assert isinstance(feedback.raw["parsed"], dict)


class TestOutputParserStripsReasoning:
    """MedGemma 1.5 reasons before answering, in Gemma's private-use tokens.

    The reasoning is prose full of category names and quoted fragments, so it
    must be removed before JSON hunting rather than merely skipped past.
    """

    @staticmethod
    def _thinking(body: str, answer: str) -> str:
        return f"<unused94>thought\n{body}<unused95>{answer}"

    def test_answer_after_a_reasoning_block_is_parsed(self) -> None:
        raw = self._thinking(
            "The category is 'pneumothorax'. Let me consider consolidation too.\n",
            json.dumps(_good_payload("cardiomegaly")),
        )
        feedback = OutputParser().parse(attempt_id="a1", raw_text=raw)
        assert feedback.raw["category"] == "cardiomegaly"

    def test_fenced_answer_after_reasoning_is_parsed(self) -> None:
        raw = self._thinking(
            "Assemble the JSON.\n",
            "```json\n" + json.dumps(_good_payload("pleural_effusion")) + "\n```",
        )
        feedback = OutputParser().parse(attempt_id="a1", raw_text=raw)
        assert feedback.raw["category"] == "pleural_effusion"

    def test_reasoning_prose_does_not_leak_into_the_category(self) -> None:
        """The reasoning names other categories; the answer must still win."""
        raw = self._thinking(
            'I first considered {"category": "pneumothorax"} but rejected it.\n',
            json.dumps(_good_payload("consolidation")),
        )
        feedback = OutputParser().parse(attempt_id="a1", raw_text=raw)
        assert feedback.raw["category"] == "consolidation"

    def test_unterminated_reasoning_block_fails_on_the_original_text(self) -> None:
        """Truncated mid-thought there is no answer to recover — fail clearly."""
        raw = "<unused94>thought\nThe image shows an absence of lung markings and"
        with pytest.raises(UpstreamUnavailableError, match="parseable JSON"):
            OutputParser().parse(attempt_id="a1", raw_text=raw)

    def test_plain_output_is_untouched(self) -> None:
        raw = json.dumps(_good_payload("fracture"))
        feedback = OutputParser().parse(attempt_id="a1", raw_text=raw)
        assert feedback.raw["category"] == "fracture"

    def test_text_merely_mentioning_the_token_is_not_treated_as_reasoning(self) -> None:
        payload = _good_payload("no_finding")
        payload["teaching_point"] = "Not a <unused95> marker."
        feedback = OutputParser().parse(attempt_id="a1", raw_text=json.dumps(payload))
        assert feedback.raw["category"] == "no_finding"


class TestDifferentialConfidenceIsBoundedByThePrimary:
    """A wrong answer cannot be more tempting than the demonstrated finding.

    The prompt asks for this and three versions failed to deliver it, with the
    violations clustering wherever the primary confidence was low. The values
    below are the ones MedGemma actually returned on the p2.0-p2.2 probe runs,
    so these tests pin real observed output rather than an imagined shape.
    """

    def _parse(self, primary: float, differentials: list[dict[str, object]]):  # type: ignore[no-untyped-def]
        payload = _good_payload()
        primary_block = payload["primary_finding"]
        assert isinstance(primary_block, dict)
        primary_block["confidence"] = primary
        payload["differential_findings"] = differentials
        return OutputParser().parse(attempt_id="a1", raw_text=json.dumps(payload))

    def test_a_differential_above_the_primary_is_pulled_down_to_it(self) -> None:
        # p2.2 case 1: 0.85 sure the heart is enlarged, "normal heart size" 0.95.
        feedback = self._parse(0.85, [{"label": "normal heart size", "confidence": 0.95}])
        assert feedback.differentials[0].confidence.value == pytest.approx(0.85)

    def test_a_differential_below_the_primary_is_untouched(self) -> None:
        feedback = self._parse(0.95, [{"label": "Pneumonia", "confidence": 0.15}])
        assert feedback.differentials[0].confidence.value == pytest.approx(0.15)

    def test_a_differential_equal_to_the_primary_is_untouched(self) -> None:
        feedback = self._parse(0.50, [{"label": "Pneumonia", "confidence": 0.50}])
        assert feedback.differentials[0].confidence.value == pytest.approx(0.50)

    def test_each_differential_is_clamped_independently(self) -> None:
        # p2.2 case 4: primary 0.50, both alternatives above it.
        feedback = self._parse(
            0.50,
            [
                {"label": "Pneumonia", "confidence": 0.80},
                {"label": "Atelectasis", "confidence": 0.70},
            ],
        )
        assert [d.confidence.value for d in feedback.differentials] == pytest.approx([0.50, 0.50])

    def test_the_label_survives_the_clamp(self) -> None:
        # Clamping must not cost the section its content: the label is the part
        # a student learns from, the number is only calibration.
        feedback = self._parse(0.50, [{"label": "Cardiomegaly", "confidence": 0.70}])
        assert feedback.differentials[0].label == "Cardiomegaly"
        assert "Cardiomegaly" in feedback.missed_clues.value

    def test_percent_scale_confidence_still_clamps_to_the_primary(self) -> None:
        # A model emitting 90 for 0.90 must not end up above a 0.5 primary.
        feedback = self._parse(0.50, [{"label": "Pneumonia", "confidence": 90}])
        assert feedback.differentials[0].confidence.value == pytest.approx(0.50)

    def test_a_zero_confidence_primary_flattens_every_differential(self) -> None:
        feedback = self._parse(0.0, [{"label": "Pneumonia", "confidence": 0.9}])
        assert feedback.differentials[0].confidence.value == pytest.approx(0.0)
