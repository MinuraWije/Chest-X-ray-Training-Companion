"""Recompute the Phase 2 metrics from a stored report, without a GPU.

    python scripts/recompute_feedback_eval.py reports/phase2_eval.json

Every input a metric needs is persisted per sample — the feedback text, the
retrieved chunk ids, the case id — so a metric that was defined wrongly can be
redefined and the whole report rebuilt offline. That is the entire reason
`FeedbackSample.to_dict` stores the feedback text: the first confabulation bug
was found after a run, and fixing it would otherwise have cost another 75
minutes of Kaggle quota to re-answer a question the data already contained.

This script does **not** reimplement any metric. It rebuilds `FeedbackSample`
objects, recomputes only the per-sample fields that depend on a changed
definition, and hands them to the same `FeedbackEvalReport` the live run uses.
A second implementation of a metric is how the two drift apart.

The model is never called, so the feedback text is exactly what the model
produced on the original run. Latency, prompt size and the model name are
carried through unchanged.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cxt.domain.entities.case import Case  # noqa: E402
from cxt.domain.entities.feedback import RetrievedPassage  # noqa: E402
from cxt.infrastructure.config.settings import get_settings  # noqa: E402
from cxt.infrastructure.data.manifest_repository import ManifestRepository  # noqa: E402
from cxt.infrastructure.eval.feedback_eval import (  # noqa: E402
    FeedbackEvalReport,
    FeedbackSample,
    _analysis_prose,
    _shuffled_control,
    _unsupported_findings,
    content_words,
    lexical_overlap,
    load_attempt_fixtures,
    render_feedback_summary,
    resolve_feedback_arm,
)
from cxt.infrastructure.retrieval.corpus import Corpus, load_corpus  # noqa: E402

logger = logging.getLogger("recompute_feedback_eval")


def _passages(chunk_ids: list[str], corpus: Corpus) -> tuple[RetrievedPassage, ...]:
    """The passages the model was handed, rebuilt from their ids.

    A chunk that has since been removed from the corpus is dropped with a
    warning rather than silently skipped: it changes what the recomputed
    numbers mean, and a reader of the new report must be told.
    """
    by_id = corpus.by_id
    out: list[RetrievedPassage] = []
    for chunk_id in chunk_ids:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            logger.warning("Chunk %s is no longer in the corpus; dropping it.", chunk_id)
            continue
        out.append(
            RetrievedPassage(text=chunk.text, source=chunk.source, score=0.0, id=chunk.id)
        )
    return tuple(out)


def _require_feedback_text(payload: dict[str, Any], path: Path) -> None:
    """Refuse to recompute a report that predates `feedback_text` persistence.

    Reports written before that change carry no feedback text, so every text
    metric recomputes to zero and the output *looks* like a result. Writing
    that over a good report destroys the run it overwrote — which is exactly
    what happened once, costing a 75-minute GPU run. Failing loudly here is the
    guard against repeating it.
    """
    for arm in payload.get("arms", ()):
        for sample in arm.get("samples", ()):
            if sample.get("ok") and not sample.get("feedback_text"):
                raise SystemExit(
                    f"{path} stores no feedback text, so its text metrics cannot be "
                    "recomputed — they would all come out as zero. It predates the "
                    "change that persists `feedback_text`. Re-run the evaluation "
                    "instead; the current code computes the metrics correctly live."
                )


def _rebuild(
    payload: dict[str, Any],
    corpus: Corpus,
    cases_by_id: dict[str, Case],
    student_text: dict[tuple[str, str], str],
) -> list[FeedbackEvalReport]:
    reports: list[FeedbackEvalReport] = []
    for arm_payload in payload["arms"]:
        arm = resolve_feedback_arm(arm_payload["arm"])
        samples: list[FeedbackSample] = []
        for offset, raw in enumerate(arm_payload["samples"]):
            passages = _passages(list(raw.get("retrieved_ids", ())), corpus)
            prose = str(raw.get("feedback_text", ""))
            analysed = _analysis_prose(prose)
            case = cases_by_id.get(raw["case_id"])
            if case is None:
                raise SystemExit(
                    f"Case {raw['case_id']} is not in the manifest; "
                    "the report cannot be recomputed against a different data set."
                )
            control = _shuffled_control(corpus, passages, offset)
            passage_words: set[str] = set()
            for passage in passages:
                passage_words |= content_words(passage.text)
            samples.append(
                FeedbackSample(
                    case_id=raw["case_id"],
                    category=raw["category"],
                    archetype=raw["archetype"],
                    ok=bool(raw.get("ok", False)),
                    latency_ms=float(raw.get("latency_ms", 0.0)),
                    retrieval_latency_ms=float(raw.get("retrieval_latency_ms", 0.0)),
                    prompt_chars=int(raw.get("prompt_chars", 0)),
                    feedback_text=prose,
                    retrieved_ids=tuple(raw.get("retrieved_ids", ())),
                    cited_indices=tuple(raw.get("cited_indices", ())),
                    hallucinated_indices=tuple(raw.get("hallucinated_indices", ())),
                    relevant_at_k=int(raw.get("relevant_at_k", 0)),
                    scored_at_k=int(raw.get("scored_at_k", 0)),
                    # The three fields whose definitions changed.
                    abstained=arm.uses_retrieval and not passages,
                    asserted_features=tuple(
                        sorted(
                            (content_words(analysed) & passage_words)
                            - content_words(
                                student_text.get((raw["category"], raw["archetype"]), "")
                            )
                        )
                    ),
                    unsupported_findings=_unsupported_findings(
                        prose, passages, case, corpus
                    ),
                    lexical_overlap=lexical_overlap(analysed, passages),
                    shuffled_overlap=lexical_overlap(analysed, control),
                    model=str(raw.get("model", "")),
                    error=str(raw.get("error", "")),
                )
            )
        reports.append(
            FeedbackEvalReport(
                arm=arm.name,
                prompt_version=arm_payload["prompt_version"],
                retrieval_mode=arm_payload["retrieval_mode"],
                corpus_version=arm_payload["corpus_version"],
                model=arm_payload["model"],
                samples=tuple(samples),
                control_arm=arm_payload.get("control_arm", ""),
            )
        )
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="path to a phase2_eval.json written by a previous run")
    parser.add_argument(
        "--out",
        default=None,
        help="where to write; defaults to overwriting the input (the .md alongside it too)",
    )
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    in_path = Path(args.report)
    if not in_path.is_absolute():
        in_path = REPO_ROOT / in_path
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    _require_feedback_text(payload, in_path)

    settings = get_settings()
    corpus = load_corpus(settings.corpus_dir)
    if corpus.version != payload.get("corpus_version"):
        logger.warning(
            "Corpus is %s but the report was produced against %s. Passage text and "
            "finding tags may differ from what the model actually saw.",
            corpus.version,
            payload.get("corpus_version"),
        )
    cases = ManifestRepository(settings.manifest_path).list(split=args.split)
    cases_by_id = {c.id: c for c in cases}
    # (category, archetype) identifies a fixture, and that is what the sample
    # records — the student's own words are needed to subtract them from the
    # vocabulary a passage supplied.
    student_text = {
        (f.category, f.archetype): f.text
        for f in load_attempt_fixtures(REPO_ROOT / "tests" / "fixtures" / "student_attempts.json")
    }

    reports = _rebuild(payload, corpus, cases_by_id, student_text)

    out_path = Path(args.out) if args.out else in_path
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.write_text(
        json.dumps(
            {
                "corpus_version": payload.get("corpus_version"),
                "split": payload.get("split", args.split),
                "max_cases": payload.get("max_cases"),
                "control_arm": payload.get("control_arm", ""),
                "recomputed_from": in_path.name,
                "arms": [r.to_dict() for r in reports],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = render_feedback_summary(reports)
    out_path.with_suffix(".md").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary)
    print(f"\nWrote {out_path} and {out_path.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
