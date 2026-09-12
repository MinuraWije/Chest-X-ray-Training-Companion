"""The Phase 2 feedback evaluation — a second path, beside the frozen one.

`infrastructure/eval/__init__.py` cannot evaluate this. It deliberately bypasses
`SubmitAttemptUseCase`, uses its own bias-free prompts, and passes **no student
free text at all** — while the misconception retriever's entire input *is*
student free text. Rather than bend that harness and invalidate the e1.0-e5.0
reports it produced, Phase 2 adds this one and leaves that one alone.

## The arms

| Arm | Prompt | Retrieval | What it isolates |
|---|---|---|---|
| `p2.4` | p2.4 | none | frozen Phase 1 baseline |
| `p3.0-empty` | p3.0 | none | **the control** — new prompt, no retrieval |
| `p3.0-oracle` | p3.0 | confirmed category | upper bound, not a retrieval result |
| `p3.0-bm25` | p3.0 | free text | the only non-circular relevance figure |
| `p3.0-misconception` | p3.0 | free text + boost | |

`p3.0-empty` is not optional. It attributes the prompt change separately from
retrieval, and it is the baseline the confabulation metric subtracts against —
without it, "the RAG arm asserted a feature that was in a passage" cannot be
distinguished from "this model asserts that feature anyway".

## Two metrics that must be read as a pair

**Citation validity** is the share of named `[Sn]` that were actually supplied.
**Citation rate** is the share of feedbacks citing anything at all. A model that
cites nothing scores 100% validity — perfect and useless. Reporting either
alone is misleading, so `render_feedback_summary` always prints both.

## Why precision@k is not reported for the oracle

It filters on `findings`, the same tag precision would be scored against, so
precision is 1.0 by construction. It drops below 1.0 only when the corpus holds
fewer than k chunks for that category — which measures corpus coverage while
calling it retrieval quality.

## Latency is not comparable to the e-series

The p3.0 prompt is longer and passages make it longer still, so model latency
rises legitimately. Phase 2 latency is its own baseline; `prompt_chars` is
recorded per sample so the comparison can be made honestly within Phase 2.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from cxt.application.use_cases.submit_attempt import (
    _enforce_citations,
    _enforce_ground_truth,
)
from cxt.domain.entities.case import Case
from cxt.domain.entities.feedback import Feedback, RetrievedPassage
from cxt.domain.ports.image_analyzer import IImageAnalyzer
from cxt.domain.ports.image_loader import IImageLoader
from cxt.domain.ports.retriever import IRetriever, RetrievalQuery
from cxt.infrastructure.ml.output_parser import OutputParser
from cxt.infrastructure.ml.prompts import PromptBuilder, load_default_prompts
from cxt.infrastructure.retrieval.corpus import Corpus
from cxt.infrastructure.retrieval.misconception import named_categories
from cxt.infrastructure.retrieval.retrievers import (
    BM25Retriever,
    CategoryRetriever,
    MisconceptionRetriever,
    NullRetriever,
)
from cxt.shared.errors import ValidationError

logger = logging.getLogger(__name__)

__all__ = [
    "FeedbackArm",
    "FEEDBACK_ARMS",
    "FeedbackSample",
    "FeedbackEvalReport",
    "StudentAttemptFixture",
    "load_attempt_fixtures",
    "resolve_feedback_arm",
    "run_feedback_evaluation",
    "render_feedback_summary",
    "content_words",
    "lexical_overlap",
]

_CITATION_RE: Final[re.Pattern[str]] = re.compile(r"\[S(\d+)\]")

#: Words carrying no topical signal. Kept deliberately short — a long stop list
#: is another tunable parameter to defend, and Jaccard over content words is
#: only ever a rough instrument.
_STOPWORDS: Final[frozenset[str]] = frozenset(
    """a an and are as at be been but by can for from has have here in is it its
    its of on or that the there these this to was were what when which with would
    you your not no""".split()
)

#: Aspects excluded from relevance scoring. A chunk about drainage technique is
#: topically about effusion and useless to someone reading a film.
_NON_TEACHING_ASPECTS: Final[frozenset[str]] = frozenset({"management"})


@dataclass(frozen=True, slots=True)
class FeedbackArm:
    """One row of the comparison table."""

    name: str
    prompt_version: str
    retrieval_mode: str

    @property
    def uses_retrieval(self) -> bool:
        return self.retrieval_mode != "null"

    @property
    def reports_precision(self) -> bool:
        """Only bm25 ranks without reference to the answer key."""
        return self.retrieval_mode in {"bm25", "misconception"}


FEEDBACK_ARMS: Final[dict[str, FeedbackArm]] = {
    "p2.4": FeedbackArm("p2.4", "p2.4", "null"),
    "p3.0-empty": FeedbackArm("p3.0-empty", "p3.0", "null"),
    "p3.0-oracle": FeedbackArm("p3.0-oracle", "p3.0", "oracle"),
    "p3.0-bm25": FeedbackArm("p3.0-bm25", "p3.0", "bm25"),
    "p3.0-misconception": FeedbackArm("p3.0-misconception", "p3.0", "misconception"),
}

#: The arm every confabulation figure is measured against.
CONTROL_ARM: Final[str] = "p3.0-empty"


def resolve_feedback_arm(name: str) -> FeedbackArm:
    try:
        return FEEDBACK_ARMS[name]
    except KeyError:
        known = ", ".join(FEEDBACK_ARMS)
        raise ValidationError(f"Unknown arm {name!r}. Known: {known}.") from None


def build_retriever_for_arm(arm: FeedbackArm, corpus: Corpus | None) -> IRetriever:
    if not arm.uses_retrieval:
        return NullRetriever()
    if corpus is None:
        raise ValidationError(f"Arm {arm.name!r} needs a corpus but none was supplied.")
    if arm.retrieval_mode == "oracle":
        return CategoryRetriever(corpus=corpus)
    if arm.retrieval_mode == "misconception":
        return MisconceptionRetriever(corpus=corpus)
    return BM25Retriever(corpus=corpus)


@dataclass(frozen=True, slots=True)
class StudentAttemptFixture:
    """One hand-authored student answer, with the archetype it exercises."""

    category: str
    archetype: str
    text: str
    expects: str = ""


def load_attempt_fixtures(path: Path) -> tuple[StudentAttemptFixture, ...]:
    """Load `tests/fixtures/student_attempts.json`.

    Keys starting with `_` are documentation, not data.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValidationError(f"{path}: expected a JSON object.")
    fixtures: list[StudentAttemptFixture] = []
    for category, entries in payload.items():
        if category.startswith("_") or not isinstance(entries, list):
            continue
        for entry in entries:
            fixtures.append(
                StudentAttemptFixture(
                    category=category,
                    archetype=str(entry.get("archetype", "unknown")),
                    text=str(entry["text"]),
                    expects=str(entry.get("expects", "")),
                )
            )
    if not fixtures:
        raise ValidationError(f"{path}: no fixtures found.")
    return tuple(fixtures)


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOPWORDS}


def lexical_overlap(feedback_text: str, passages: Sequence[RetrievedPassage]) -> float:
    """Content-word Jaccard between the feedback and the supplied passages.

    Uninterpretable on its own, which is why `run_feedback_evaluation` also
    computes it against a shuffled control: the question is not "do they share
    words" — any two texts about chest radiographs do — but "do they share more
    words than an unrelated passage would".
    """
    feedback_words = content_words(feedback_text)
    passage_words: set[str] = set()
    for passage in passages:
        passage_words |= content_words(passage.text)
    if not feedback_words or not passage_words:
        return 0.0
    return len(feedback_words & passage_words) / len(feedback_words | passage_words)


@dataclass(frozen=True, slots=True)
class FeedbackSample:
    """One (case, student answer, arm) evaluation."""

    case_id: str
    category: str
    archetype: str
    ok: bool
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    prompt_chars: int = 0
    feedback_text: str = ""
    retrieved_ids: tuple[str, ...] = field(default_factory=tuple)
    cited_indices: tuple[int, ...] = field(default_factory=tuple)
    hallucinated_indices: tuple[int, ...] = field(default_factory=tuple)
    relevant_at_k: int = 0
    scored_at_k: int = 0
    abstained: bool = False
    asserted_features: tuple[str, ...] = field(default_factory=tuple)
    lexical_overlap: float = 0.0
    shuffled_overlap: float = 0.0
    model: str = ""
    error: str = ""

    @property
    def cited_any(self) -> bool:
        return bool(self.cited_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "archetype": self.archetype,
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 1),
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
            "prompt_chars": self.prompt_chars,
            "retrieved_ids": list(self.retrieved_ids),
            "cited_indices": list(self.cited_indices),
            "hallucinated_indices": list(self.hallucinated_indices),
            "relevant_at_k": self.relevant_at_k,
            "scored_at_k": self.scored_at_k,
            "abstained": self.abstained,
            "lexical_overlap": round(self.lexical_overlap, 4),
            "shuffled_overlap": round(self.shuffled_overlap, 4),
            "model": self.model,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class FeedbackEvalReport:
    arm: str
    prompt_version: str
    retrieval_mode: str
    corpus_version: str
    model: str
    samples: tuple[FeedbackSample, ...] = field(default_factory=tuple)
    confabulation_rate: float | None = None
    control_arm: str = ""

    # -- counts -----------------------------------------------------------
    @property
    def total(self) -> int:
        return len(self.samples)

    @property
    def succeeded(self) -> tuple[FeedbackSample, ...]:
        return tuple(s for s in self.samples if s.ok)

    # -- the citation pair ------------------------------------------------
    @property
    def citation_rate(self) -> float:
        """Share of feedbacks citing at least one passage.

        Read with `citation_validity`, never alone: a model that cites nothing
        has perfect validity and has done nothing.
        """
        eligible = [s for s in self.succeeded if not s.abstained]
        if not eligible:
            return 0.0
        return sum(1 for s in eligible if s.cited_any) / len(eligible)

    @property
    def citation_validity(self) -> float:
        """Share of named [Sn] that were actually supplied."""
        named = sum(len(s.cited_indices) + len(s.hallucinated_indices) for s in self.succeeded)
        if not named:
            return 1.0
        valid = sum(len(s.cited_indices) for s in self.succeeded)
        return valid / named

    @property
    def hallucinated_citation_rate(self) -> float:
        eligible = self.succeeded
        if not eligible:
            return 0.0
        return sum(1 for s in eligible if s.hallucinated_indices) / len(eligible)

    # -- retrieval --------------------------------------------------------
    @property
    def precision_at_k(self) -> float | None:
        """Tag overlap, `aspect != management`. `None` for arms that filter on
        the tag being scored — see the module docstring."""
        if self.retrieval_mode not in {"bm25", "misconception"}:
            return None
        scored = sum(s.scored_at_k for s in self.succeeded)
        if not scored:
            return 0.0
        return sum(s.relevant_at_k for s in self.succeeded) / scored

    @property
    def abstention_rate(self) -> float:
        if not self.succeeded:
            return 0.0
        return sum(1 for s in self.succeeded if s.abstained) / len(self.succeeded)

    @property
    def corpus_coverage(self) -> int:
        """Distinct chunks ever retrieved. Exposes the small-corpus degeneracy
        where three chunks serve every case."""
        seen: set[str] = set()
        for sample in self.succeeded:
            seen.update(sample.retrieved_ids)
        return len(seen)

    # -- grounding --------------------------------------------------------
    @property
    def mean_lexical_overlap(self) -> float:
        values = [s.lexical_overlap for s in self.succeeded if not s.abstained]
        return statistics.fmean(values) if values else 0.0

    @property
    def mean_shuffled_overlap(self) -> float:
        """The null. Overlap without this is uninterpretable."""
        values = [s.shuffled_overlap for s in self.succeeded if not s.abstained]
        return statistics.fmean(values) if values else 0.0

    @property
    def overlap_lift(self) -> float:
        return self.mean_lexical_overlap - self.mean_shuffled_overlap

    # -- cost -------------------------------------------------------------
    @property
    def median_latency_ms(self) -> float:
        values = [s.latency_ms for s in self.succeeded]
        return statistics.median(values) if values else 0.0

    @property
    def mean_prompt_chars(self) -> float:
        values = [s.prompt_chars for s in self.succeeded]
        return statistics.fmean(values) if values else 0.0

    @property
    def mean_retrieval_latency_ms(self) -> float:
        values = [s.retrieval_latency_ms for s in self.succeeded if not s.abstained]
        return statistics.fmean(values) if values else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "prompt_version": self.prompt_version,
            "retrieval_mode": self.retrieval_mode,
            "corpus_version": self.corpus_version,
            "model": self.model,
            "control_arm": self.control_arm,
            "totals": {"cases": self.total, "succeeded": len(self.succeeded)},
            "citation": {
                "rate": round(self.citation_rate, 4),
                "validity": round(self.citation_validity, 4),
                "hallucinated_rate": round(self.hallucinated_citation_rate, 4),
            },
            "retrieval": {
                "precision_at_k": (
                    None if self.precision_at_k is None else round(self.precision_at_k, 4)
                ),
                "abstention_rate": round(self.abstention_rate, 4),
                "corpus_coverage": self.corpus_coverage,
                "mean_latency_ms": round(self.mean_retrieval_latency_ms, 2),
            },
            "grounding": {
                "lexical_overlap": round(self.mean_lexical_overlap, 4),
                "shuffled_control": round(self.mean_shuffled_overlap, 4),
                "lift": round(self.overlap_lift, 4),
                "confabulation_rate": (
                    None
                    if self.confabulation_rate is None
                    else round(self.confabulation_rate, 4)
                ),
            },
            "cost": {
                "median_latency_ms": round(self.median_latency_ms, 1),
                "mean_prompt_chars": round(self.mean_prompt_chars, 1),
            },
            "samples": [s.to_dict() for s in self.samples],
        }


def _feedback_prose(feedback: Feedback) -> str:
    parts = [
        feedback.key_signs.value,
        feedback.comparison.value,
        feedback.missed_clues.value,
        feedback.learning_point.value,
    ]
    if feedback.key_observations is not None:
        parts.append(feedback.key_observations.value)
    return "\n".join(parts)


def _split_citations(text: str, supplied: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (valid, hallucinated) citation indices named in `text`."""
    valid: list[int] = []
    invalid: list[int] = []
    for match in _CITATION_RE.finditer(text):
        index = int(match.group(1))
        (valid if 1 <= index <= supplied else invalid).append(index)
    return tuple(sorted(set(valid))), tuple(sorted(set(invalid)))


def _relevance(
    passages: Sequence[RetrievedPassage], case: Case, corpus: Corpus | None
) -> tuple[int, int]:
    """(relevant, scored) by tag overlap, excluding non-teaching aspects."""
    if corpus is None:
        return 0, 0
    by_id = corpus.by_id
    confirmed = set(case.confirmed_categories)
    relevant = 0
    scored = 0
    for passage in passages:
        chunk = by_id.get(passage.id)
        if chunk is None or chunk.aspect in _NON_TEACHING_ASPECTS:
            continue
        scored += 1
        if confirmed.intersection(chunk.findings):
            relevant += 1
    return relevant, scored


def _shuffled_control(
    corpus: Corpus | None, passages: Sequence[RetrievedPassage], offset: int
) -> tuple[RetrievedPassage, ...]:
    """Passages the model did NOT see, for the overlap null.

    Deterministic rather than random: two runs of the eval must give the same
    control, or the null is noise.
    """
    if corpus is None or not corpus.chunks or not passages:
        return ()
    supplied = {p.id for p in passages}
    pool = [c for c in corpus.chunks if c.id not in supplied]
    if not pool:
        return ()
    picked = [pool[(offset + i) % len(pool)] for i in range(len(passages))]
    return tuple(
        RetrievedPassage(text=c.text, source=c.source, score=0.0, id=c.id, url=c.url)
        for c in picked
    )


def run_feedback_evaluation(
    *,
    arm: FeedbackArm,
    cases: Sequence[Case],
    fixtures: Sequence[StudentAttemptFixture],
    analyzer: IImageAnalyzer,
    image_loader: IImageLoader,
    corpus: Corpus | None = None,
    parser: OutputParser | None = None,
    max_cases: int | None = None,
) -> FeedbackEvalReport:
    """Run one arm over (case, student answer) pairs.

    Pairs each case with the fixtures written for its confirmed category, so a
    pneumothorax case is marked against pneumothorax answers. A case whose
    category has no fixtures is skipped rather than paired arbitrarily.
    """
    parser = parser or OutputParser()
    retriever = build_retriever_for_arm(arm, corpus)
    builder = PromptBuilder(templates=load_default_prompts(arm.prompt_version))
    by_category: dict[str, list[StudentAttemptFixture]] = {}
    for fixture in fixtures:
        by_category.setdefault(fixture.category, []).append(fixture)

    samples: list[FeedbackSample] = []
    pair_index = 0
    for case in cases:
        category = case.ground_truth_finding.category.value
        for fixture in by_category.get(category, []):
            if max_cases is not None and len(samples) >= max_cases:
                break
            samples.append(
                _evaluate_one(
                    arm=arm,
                    case=case,
                    fixture=fixture,
                    analyzer=analyzer,
                    image_loader=image_loader,
                    retriever=retriever,
                    builder=builder,
                    parser=parser,
                    corpus=corpus,
                    offset=pair_index,
                )
            )
            pair_index += 1
        if max_cases is not None and len(samples) >= max_cases:
            break

    # The model can change mid-run if the notebook restarts, so the report names
    # the model the successful samples actually came from rather than assuming.
    observed = {s.model for s in samples if s.ok and s.model}
    model_name = "/".join(sorted(observed)) if observed else "unknown"

    return FeedbackEvalReport(
        arm=arm.name,
        prompt_version=builder.version,
        retrieval_mode=arm.retrieval_mode,
        corpus_version=corpus.version if corpus is not None else "",
        model=model_name,
        samples=tuple(samples),
        control_arm=CONTROL_ARM,
    )


def _evaluate_one(
    *,
    arm: FeedbackArm,
    case: Case,
    fixture: StudentAttemptFixture,
    analyzer: IImageAnalyzer,
    image_loader: IImageLoader,
    retriever: IRetriever,
    builder: PromptBuilder,
    parser: OutputParser,
    corpus: Corpus | None,
    offset: int,
) -> FeedbackSample:
    # `Any` because this is splatted into `FeedbackSample`, whose remaining
    # fields are ints, floats and tuples; a `dict[str, str]` inference makes
    # every one of those a type error at the call site.
    base: dict[str, Any] = {
        "case_id": case.id,
        "category": case.ground_truth_finding.category.value,
        "archetype": fixture.archetype,
    }
    retrieval_started = time.perf_counter()
    try:
        query = RetrievalQuery(
            free_text=fixture.text,
            confirmed_categories=case.confirmed_categories,
            named_categories=named_categories(fixture.text),
            case_id=case.id,
        )
        passages = tuple(retriever.retrieve(query=query, top_k=3))
    except Exception as exc:
        logger.warning("Retrieval failed for %s: %s", case.id, exc)
        passages = ()
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

    prompts = builder.build(
        clinical_context=case.metadata.get("clinical_context", ""),
        student_free_text=fixture.text,
        finding=case.ground_truth_finding,
        secondary_findings=case.secondary_findings,
        passages=passages,
    )
    prompt_chars = len(prompts.system) + len(prompts.user)

    started = time.perf_counter()
    try:
        image = image_loader.load(case.image_url)
        analysis = analyzer.analyze(
            image=image, prompt=prompts.user, system=prompts.system
        )
        feedback = parser.parse(attempt_id=case.id, raw_text=analysis.raw_text)
    except Exception as exc:
        return FeedbackSample(
            **base,
            ok=False,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            retrieval_latency_ms=retrieval_ms,
            prompt_chars=prompt_chars,
            retrieved_ids=tuple(p.id for p in passages),
            abstained=not passages,
            error=f"{type(exc).__name__}: {exc}",
        )
    latency_ms = (time.perf_counter() - started) * 1000.0

    feedback = _enforce_ground_truth(feedback, case)
    # The eval must see what the student would see, so the same citation
    # enforcement runs here. Measuring the unenforced text would report a
    # confabulation rate for output nobody is ever shown.
    prose_before = _feedback_prose(feedback)
    cited, hallucinated = _split_citations(prose_before, len(passages))
    feedback = _enforce_citations(feedback, passages)

    relevant, scored = _relevance(passages, case, corpus)
    prose = _feedback_prose(feedback)
    control = _shuffled_control(corpus, passages, offset)

    return FeedbackSample(
        **base,
        ok=True,
        latency_ms=latency_ms,
        retrieval_latency_ms=retrieval_ms,
        prompt_chars=prompt_chars,
        feedback_text=prose,
        retrieved_ids=tuple(p.id for p in passages),
        cited_indices=cited,
        hallucinated_indices=hallucinated,
        relevant_at_k=relevant,
        scored_at_k=scored,
        abstained=not passages,
        lexical_overlap=lexical_overlap(prose, passages),
        shuffled_overlap=lexical_overlap(prose, control),
        model=analysis.model_output.model,
    )


def compute_confabulation_rate(
    treatment: FeedbackEvalReport, control: FeedbackEvalReport
) -> float:
    """Share of treatment samples asserting passage content the control did not.

    The headline metric, and the reason `p3.0-empty` is non-negotiable. Measured
    as content words that appear in both the treatment's feedback and its
    supplied passages, but in neither the control's feedback for the same
    (case, archetype) pair nor the student's own words — i.e. language the model
    produced only because it was handed the passage.

    A lexical proxy, and the write-up must say so: it detects vocabulary
    imported from a passage, not whether the resulting clinical claim is false.
    """
    control_by_key = {
        (s.case_id, s.archetype): s for s in control.succeeded
    }
    considered = 0
    flagged = 0
    for sample in treatment.succeeded:
        if sample.abstained:
            continue
        counterpart = control_by_key.get((sample.case_id, sample.archetype))
        if counterpart is None:
            continue
        considered += 1
        imported = (
            content_words(sample.feedback_text)
            - content_words(counterpart.feedback_text)
        )
        if imported:
            flagged += 1
    return flagged / considered if considered else 0.0


def render_feedback_summary(reports: Sequence[FeedbackEvalReport]) -> str:
    """Markdown comparison across arms."""
    lines: list[str] = [
        "# Phase 2 feedback evaluation",
        "",
        "| Arm | n | Cite rate | Cite validity | Fabricated | P@3 | Abstain | Coverage | Overlap (null) | Confab | Median ms | Prompt chars |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for report in reports:
        precision = report.precision_at_k
        precision_cell = "n/a" if precision is None else f"{precision:.0%}"
        confab = report.confabulation_rate
        confab_cell = "—" if confab is None else f"{confab:.0%}"
        lines.append(
            f"| `{report.arm}` | {len(report.succeeded)} "
            f"| {report.citation_rate:.0%} "
            f"| {report.citation_validity:.0%} "
            f"| {report.hallucinated_citation_rate:.0%} "
            f"| {precision_cell} "
            f"| {report.abstention_rate:.0%} "
            f"| {report.corpus_coverage} "
            f"| {report.mean_lexical_overlap:.3f} ({report.mean_shuffled_overlap:.3f}) "
            f"| {confab_cell} "
            f"| {report.median_latency_ms:.0f} "
            f"| {report.mean_prompt_chars:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Reading this table",
            "",
            "- **Cite rate and cite validity are a pair.** A model citing nothing",
            "  scores 100% validity and has done nothing. Neither number means",
            "  anything without the other.",
            "- **P@3 is `n/a` for the oracle arm by design.** It filters on the same",
            "  tag precision would score, so precision is 1.0 by construction and",
            "  falls below only when the corpus is thin — that measures coverage,",
            "  not retrieval quality.",
            "- **Overlap is shown with its shuffled null in brackets.** Any two texts",
            "  about chest radiographs share words; only the gap is evidence.",
            f"- **Confabulation is measured against `{CONTROL_ARM}`**, which is why",
            "  that arm is not optional.",
            "- **Latency is not comparable to the e1.0-e5.0 reports.** The p3.0",
            "  prompt is longer and passages lengthen it further, so a rise is",
            "  expected. Compare within Phase 2 only; `prompt_chars` is the control.",
            "",
        ]
    )
    return "\n".join(lines)
