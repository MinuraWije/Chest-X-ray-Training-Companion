"""Retrieval inside `SubmitAttemptUseCase`: ordering, citations, abstention.

The ordering test is the important one. In Phase 1 retrieval ran *after*
generation and the passages were stapled onto the finished feedback, so they
grounded nothing — a grounding metric over a post-hoc attachment measures only
whether prose written in ignorance happens to agree with a document it never
saw. If that regressed, every Phase 2 number would still compute and would all
be meaningless, which is precisely the kind of failure a test has to catch
because nothing in the output looks wrong.
"""
from __future__ import annotations

import json

from cxt.application.use_cases.submit_attempt import (
    SubmitAttemptRequest,
    SubmitAttemptUseCase,
)
from cxt.domain.entities.finding import FindingCategory
from cxt.infrastructure.ml.prompts import PromptBuilder, load_default_prompts
from tests.support.doubles import (
    StubAnalyzer,
    StubCaseRepo,
    StubRetriever,
    feedback_payload,
    make_case,
    make_passages,
)


class RecordingAttemptRepo:
    """IAttemptRepository that keeps what it was handed."""

    def __init__(self) -> None:
        self.attempts: list[object] = []
        self.feedback: dict[str, object] = {}

    def add(self, attempt: object) -> None:
        self.attempts.append(attempt)

    def attach_feedback(self, attempt_id: str, feedback: object) -> None:
        self.feedback[attempt_id] = feedback

    def get_with_feedback(self, attempt_id: str) -> tuple[object, object | None]:
        raise NotImplementedError

    def list_for_student(self, student_id: str) -> list[object]:
        return []

    def list_for_educator_review(self) -> list[object]:
        return []


def _build(
    *,
    retriever: StubRetriever,
    raw_text: str | None = None,
    category: str = "pleural_effusion",
    prompt_version: str = "p3.0",
    **kwargs: object,
) -> tuple[SubmitAttemptUseCase, StubAnalyzer, RecordingAttemptRepo]:
    case = make_case("case-1", FindingCategory.PLEURAL_EFFUSION)
    analyzer = StubAnalyzer(default_category=category, raw_text=raw_text)
    attempts = RecordingAttemptRepo()
    use_case = SubmitAttemptUseCase(
        analyzer=analyzer,
        attempt_repository=attempts,  # type: ignore[arg-type]
        case_repository=StubCaseRepo([case]),
        retriever=retriever,  # type: ignore[arg-type]
        prompt_builder=PromptBuilder(templates=load_default_prompts(prompt_version)),
        **kwargs,  # type: ignore[arg-type]
    )
    return use_case, analyzer, attempts


def _run(use_case: SubmitAttemptUseCase, free_text: str = "I think there is fluid at the right base.") -> object:
    return use_case.execute(
        SubmitAttemptRequest(case_id="case-1", student_id="stu-1", free_text=free_text)
    )


class TestRetrievalHappensBeforeGeneration:
    def test_passages_reach_the_prompt(self) -> None:
        retriever = StubRetriever(make_passages(2))
        use_case, analyzer, _ = _build(retriever=retriever)
        _run(use_case)
        _, prompt, _system = analyzer.calls[0]
        assert "[S1] Teaching passage number 1." in prompt
        assert "[S2] Teaching passage number 2." in prompt

    def test_retriever_is_called_before_the_analyzer(self) -> None:
        retriever = StubRetriever(make_passages(1))
        use_case, analyzer, _ = _build(retriever=retriever)
        _run(use_case)
        # The analyzer cannot have been called with passages in its prompt
        # unless retrieval preceded it; asserting on the prompt is the
        # strongest available ordering evidence.
        assert retriever.queries
        assert "[S1]" in analyzer.calls[0][1]

    def test_query_carries_the_signals_each_arm_needs(self) -> None:
        retriever = StubRetriever(make_passages(1))
        use_case, _, _ = _build(retriever=retriever)
        _run(use_case, "I think this is a pleural effusion, not a pneumothorax.")
        query = retriever.queries[0]
        assert query.case_id == "case-1"
        assert "pleural_effusion" in query.confirmed_categories
        assert "pleural_effusion" in query.named_categories
        # "not a pneumothorax" is a correct exclusion, not a claim.
        assert "pneumothorax" not in query.named_categories

    def test_sources_record_what_was_supplied(self) -> None:
        passages = make_passages(3)
        use_case, _, _ = _build(retriever=StubRetriever(passages))
        result = _run(use_case)
        assert [s.id for s in result.feedback.sources] == [  # type: ignore[attr-defined]
            p.id for p in passages
        ]


class TestAbstention:
    def test_no_passages_renders_the_explicit_absence(self) -> None:
        use_case, analyzer, _ = _build(retriever=StubRetriever(()))
        _run(use_case)
        assert "No reference material was supplied" in analyzer.calls[0][1]

    def test_abstention_is_recorded_as_a_decision(self) -> None:
        use_case, _, _ = _build(retriever=StubRetriever(()))
        result = _run(use_case)
        assert result.feedback.raw["abstained"] is True  # type: ignore[attr-defined]
        assert result.feedback.sources == ()  # type: ignore[attr-defined]

    def test_supplying_passages_clears_the_abstention_flag(self) -> None:
        use_case, _, _ = _build(retriever=StubRetriever(make_passages(1)))
        result = _run(use_case)
        assert result.feedback.raw["abstained"] is False  # type: ignore[attr-defined]


class TestDegradation:
    def test_retriever_failure_does_not_cost_the_student_their_attempt(self) -> None:
        """A broken corpus index degrades to Phase 1, it does not 500."""
        retriever = StubRetriever(raises=RuntimeError("index corrupt"))
        use_case, _, _ = _build(retriever=retriever)
        result = _run(use_case)
        assert result.feedback is not None  # type: ignore[attr-defined]
        assert result.feedback.sources == ()  # type: ignore[attr-defined]

    def test_failure_is_reported_as_an_abstention_not_a_silent_success(self) -> None:
        retriever = StubRetriever(raises=RuntimeError("index corrupt"))
        use_case, _, _ = _build(retriever=retriever)
        result = _run(use_case)
        assert result.feedback.raw["abstained"] is True  # type: ignore[attr-defined]


class TestCitationEnforcement:
    def _payload_with(self, text: str) -> str:
        payload = feedback_payload("pleural_effusion")
        payload["key_observations"] = text  # type: ignore[index]
        return json.dumps(payload)

    def test_a_fabricated_citation_is_stripped(self) -> None:
        """The model was shown two passages and cites a fourth."""
        raw = self._payload_with("Blunting is visible [S1]. A meniscus rises [S4].")
        use_case, _, _ = _build(retriever=StubRetriever(make_passages(2)), raw_text=raw)
        result = _run(use_case)
        observations = result.feedback.key_observations.value  # type: ignore[attr-defined]
        assert "[S1]" in observations
        assert "[S4]" not in observations

    def test_the_removal_is_recorded_for_the_eval(self) -> None:
        raw = self._payload_with("A meniscus rises [S4].")
        use_case, _, _ = _build(retriever=StubRetriever(make_passages(2)), raw_text=raw)
        result = _run(use_case)
        assert result.feedback.raw["citations_stripped"] == ["[S4]"]  # type: ignore[attr-defined]

    def test_valid_citations_are_left_alone(self) -> None:
        raw = self._payload_with("Blunting is visible [S1] and a meniscus [S2].")
        use_case, _, _ = _build(retriever=StubRetriever(make_passages(3)), raw_text=raw)
        result = _run(use_case)
        observations = result.feedback.key_observations.value  # type: ignore[attr-defined]
        assert "[S1]" in observations and "[S2]" in observations
        assert "citations_stripped" not in result.feedback.raw  # type: ignore[attr-defined]

    def test_every_citation_is_stripped_when_nothing_was_supplied(self) -> None:
        raw = self._payload_with("As described in the literature [S1].")
        use_case, _, _ = _build(retriever=StubRetriever(()), raw_text=raw)
        result = _run(use_case)
        assert "[S1]" not in result.feedback.key_observations.value  # type: ignore[attr-defined]

    def test_a_stripped_field_never_becomes_empty(self) -> None:
        """Rationale rejects empty strings; a feedback field that was nothing
        but a citation must not blow up the whole submission."""
        raw = self._payload_with("[S9]")
        use_case, _, _ = _build(retriever=StubRetriever(()), raw_text=raw)
        result = _run(use_case)
        assert result.feedback.key_observations.value.strip()  # type: ignore[attr-defined]


class TestProvenance:
    def test_records_the_arm_and_the_corpus(self) -> None:
        """Two runs are only comparable if these match."""
        use_case, _, _ = _build(
            retriever=StubRetriever(make_passages(2)),
            retrieval_mode="bm25",
            corpus_version="c1.23.abc123",
        )
        raw = _run(use_case).feedback.raw  # type: ignore[attr-defined]
        assert raw["retrieval_mode"] == "bm25"
        assert raw["corpus_version"] == "c1.23.abc123"
        assert raw["retrieval_k"] == 2
        assert raw["prompt_version"].startswith("p3.0+")

    def test_records_the_ranked_chunk_ids(self) -> None:
        use_case, _, _ = _build(retriever=StubRetriever(make_passages(2)))
        raw = _run(use_case).feedback.raw  # type: ignore[attr-defined]
        assert [row["id"] for row in raw["retrieval"]] == ["chunk-1", "chunk-2"]
        assert [row["rank"] for row in raw["retrieval"]] == [1, 2]


class TestBaselineArmIsUnaffected:
    def test_p24_ignores_supplied_passages(self) -> None:
        """The frozen Phase 1 arm must render identically whatever Phase 2 does."""
        use_case, analyzer, _ = _build(
            retriever=StubRetriever(make_passages(3)), prompt_version="p2.4"
        )
        _run(use_case)
        prompt = analyzer.calls[0][1]
        assert "[S1]" not in prompt
        assert "No reference material" not in prompt
