"""Prompt template loading, the evidence block, and the notebook's size caps.

The size caps are the reason this file exists. `kaggle_runtime/medgemma_server.py`
declares `prompt: str = Field(max_length=8000)` and `system: str = Field(max_length=4000)`,
so an over-long prompt is a 422 from the notebook rather than a worse answer —
and it happens on the subset of cases with the most metadata, which is not the
subset anyone probes first. The p2.4 baseline currently renders within three
characters of its cap. Without an assertion, the next prompt edit ships broken.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cxt.domain.entities.feedback import RetrievedPassage
from cxt.domain.entities.finding import Finding, FindingCategory
from cxt.domain.value_objects.finding_label import FindingLabel
from cxt.infrastructure.config.settings import get_settings
from cxt.infrastructure.data.manifest_repository import (
    MAX_CLINICAL_CONTEXT_CHARS,
    ManifestRepository,
    build_clinical_context,
)
from cxt.infrastructure.ml.prompts import (
    KNOWN_PROMPT_VERSIONS,
    PROMPT_VERSION,
    PromptBuilder,
    load_default_prompts,
    render_evidence_block,
)
from cxt.shared.errors import ValidationError

# Mirrors kaggle_runtime/medgemma_server.py:86-87.
SYSTEM_CAP = 4000
PROMPT_CAP = 8000

# The longest input the API will accept (attempts.py `_MAX_FREE_TEXT`) and the
# corpus chunk ceiling. Budgets are only meaningful at the worst case.
MAX_FREE_TEXT = 2000
MAX_CHUNK_CHARS = 700


def _worst_case_passages(k: int = 3) -> tuple[RetrievedPassage, ...]:
    return tuple(
        RetrievedPassage(text="w" * MAX_CHUNK_CHARS, source="Source", id=f"chunk-{i}")
        for i in range(k)
    )


def _a_finding() -> Finding:
    return Finding(
        category=FindingCategory.PLEURAL_EFFUSION,
        label=FindingLabel("Right pleural effusion"),
    )


class TestPromptVersions:
    def test_every_known_version_has_both_templates(self) -> None:
        for version in KNOWN_PROMPT_VERSIONS:
            templates = load_default_prompts(version)
            assert templates.system.strip()
            assert templates.user.strip()
            assert templates.version == version

    def test_default_is_the_current_version(self) -> None:
        assert load_default_prompts().version == PROMPT_VERSION

    def test_unknown_version_raises_rather_than_defaulting(self) -> None:
        """A silent fallback would let a mistyped arm be reported as a real one."""
        with pytest.raises(ValidationError, match="Unknown prompt version"):
            load_default_prompts("p9.9")

    def test_p24_is_still_the_frozen_baseline(self) -> None:
        """p2.4's text is the Phase 1 arm. Editing it invalidates the e-series."""
        system = load_default_prompts("p2.4").system
        assert "Return a JSON object with EXACTLY these keys" in system
        assert "$evidence_block" not in system

    def test_version_string_names_the_loaded_templates(self) -> None:
        builder = PromptBuilder(templates=load_default_prompts("p2.4"))
        assert builder.version.startswith("p2.4+")


class TestEvidenceBlock:
    def test_no_passages_renders_an_explicit_absence(self) -> None:
        """The control arm is this call. A blank gap would not be a control."""
        block = render_evidence_block(())
        assert "No reference material was supplied" in block
        assert "[S1]" not in block

    def test_passages_are_labelled_positionally_and_opaquely(self) -> None:
        passages = (
            RetrievedPassage(
                text="Blunting of the costophrenic angle.",
                source="WHO p.42",
                id="who-2006-p42-effusion",
            ),
            RetrievedPassage(
                text="A meniscus rises laterally.",
                source="WHO p.43",
                id="who-2006-p43-meniscus",
            ),
        )
        block = render_evidence_block(passages)
        assert "[S1] Blunting of the costophrenic angle." in block
        assert "[S2] A meniscus rises laterally." in block
        assert "[S3]" not in block
        # The model is told what a passage says, never whose authority it
        # carries: naming the source invites it to weigh a textbook against the
        # film. Labels map back to ids via `Feedback.sources`, in order.
        assert "WHO p.42" not in block
        assert "who-2006-p42-effusion" not in block

    def test_label_instruction_names_only_the_labels_supplied(self) -> None:
        block = render_evidence_block(_worst_case_passages(2))
        assert "[S1], [S2] only where" in block
        assert "[S3]" not in block


class TestClinicalContext:
    def test_renders_the_demographics_the_manifest_holds(self) -> None:
        rendered = build_clinical_context(
            {"patient_age": "56", "patient_sex": "M", "view_position": "AP"}
        )
        assert rendered == "56-year-old man, AP (supine/portable) film."

    def test_empty_row_falls_back_to_the_prompt_sentinel(self) -> None:
        assert build_clinical_context({}) == ""
        built = PromptBuilder(templates=load_default_prompts()).build(
            clinical_context="",
            student_free_text="Looks clear.",
            finding=_a_finding(),
        )
        assert "(no clinical context provided)" in built.system

    def test_never_exceeds_the_budget_that_p24_depends_on(self) -> None:
        """Every manifest row, not just the tidy ones."""
        repo = ManifestRepository(get_settings().manifest_path)
        for case in repo.list():
            rendered = case.metadata["clinical_context"]
            assert len(rendered) <= MAX_CLINICAL_CONTEXT_CHARS, case.id

    def test_absurd_row_is_trimmed_rather_than_overflowing(self) -> None:
        rendered = build_clinical_context(
            {"patient_age": "56", "patient_sex": "M", "view_position": "X" * 200}
        )
        assert len(rendered) <= MAX_CLINICAL_CONTEXT_CHARS


class TestPromptBudget:
    """Rendered prompts must fit the notebook's caps for every real case."""

    @pytest.mark.parametrize("version", KNOWN_PROMPT_VERSIONS)
    def test_worst_case_manifest_row_fits(self, version: str) -> None:
        manifest = get_settings().manifest_path
        if not Path(manifest).exists():  # pragma: no cover - real data present in CI
            pytest.skip("curated manifest not built")
        repo = ManifestRepository(manifest)
        builder = PromptBuilder(templates=load_default_prompts(version))
        passages = _worst_case_passages()
        worst_system = 0
        worst_user = 0
        for case in repo.list():
            rendered = builder.build(
                clinical_context=case.metadata.get("clinical_context", ""),
                student_free_text="w" * MAX_FREE_TEXT,
                finding=case.ground_truth_finding,
                secondary_findings=case.secondary_findings,
                passages=passages,
            )
            worst_system = max(worst_system, len(rendered.system))
            worst_user = max(worst_user, len(rendered.user))
        assert worst_system <= SYSTEM_CAP, (
            f"{version} system prompt renders at {worst_system} chars against a "
            f"{SYSTEM_CAP} cap. The notebook will answer 422, not worse feedback."
        )
        assert worst_user <= PROMPT_CAP, (
            f"{version} user prompt renders at {worst_user} chars against a {PROMPT_CAP} cap."
        )

    def test_passages_reach_the_user_turn_not_the_system_turn(self) -> None:
        builder = PromptBuilder(templates=load_default_prompts("p3.0"))
        rendered = builder.build(
            clinical_context="56-year-old man, AP (supine/portable) film.",
            student_free_text="I think the right base is opacified.",
            finding=_a_finding(),
            passages=(RetrievedPassage(text="A meniscus sign.", source="WHO", id="x"),),
        )
        assert "[S1] A meniscus sign." in rendered.user
        # The system turn names the labels in a rule; what must never appear
        # there is passage *content*, which is what eats the 4000-char budget.
        assert "A meniscus sign." not in rendered.system

    def test_p24_ignores_passages_entirely(self) -> None:
        """The frozen arm must be unaffected by anything Phase 2 passes it."""
        builder = PromptBuilder(templates=load_default_prompts("p2.4"))
        common = {
            "clinical_context": "56-year-old man, AP (supine/portable) film.",
            "student_free_text": "Right base opacified.",
            "finding": _a_finding(),
        }
        without = builder.build(**common)  # type: ignore[arg-type]
        with_passages = builder.build(**common, passages=_worst_case_passages())  # type: ignore[arg-type]
        assert without.system == with_passages.system
        assert without.user == with_passages.user
