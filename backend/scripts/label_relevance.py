"""Export retrieval pairs for manual relevance labelling, then score the proxy.

    python scripts/label_relevance.py export --cases 30 --top-k 5
    # ... a human fills in `label` on each pair in data/eval/relevance_labels.json
    python scripts/label_relevance.py validate

`export` needs no model and no GPU: retrieval is deterministic and runs
offline, so the labelling sheet can be produced and filled in while the Kaggle
notebook is unavailable. That is deliberate — this is the one Phase 2
measurement that does not compete for GPU quota.

The point of the exercise is stated in `infrastructure/eval/relevance_labels.py`:
precision@3 is computed from tag overlap, which is cheap and obviously
incomplete, and "my automatic metric agrees with my manual labels on 82% of
pairs" is a far stronger claim than the precision figure alone.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cxt.domain.entities.case import Case  # noqa: E402
from cxt.domain.ports.retriever import IRetriever, RetrievalQuery  # noqa: E402
from cxt.infrastructure.config.settings import get_settings  # noqa: E402
from cxt.infrastructure.data.manifest_repository import ManifestRepository  # noqa: E402
from cxt.infrastructure.eval.feedback_eval import (  # noqa: E402
    StudentAttemptFixture,
    build_retriever_for_arm,
    load_attempt_fixtures,
    resolve_feedback_arm,
)
from cxt.infrastructure.eval.relevance_labels import (  # noqa: E402
    LabelledPair,
    export_labelling_sheet,
    load_labels,
    render_agreement_summary,
    validate_proxy,
)
from cxt.infrastructure.retrieval.corpus import Corpus, load_corpus  # noqa: E402
from cxt.infrastructure.retrieval.misconception import named_categories  # noqa: E402

logger = logging.getLogger("label_relevance")

FIXTURES_PATH = REPO_ROOT / "tests" / "fixtures" / "student_attempts.json"
DEFAULT_LABELS_PATH = REPO_ROOT.parent / "data" / "eval" / "relevance_labels.json"

#: Aspects the report already excludes from relevance. Kept in the sheet so a
#: labeller can confirm the exclusion is doing what it claims, rather than
#: having to trust it.
_NON_TEACHING = {"management"}


def _sample_queries(
    cases: list[Case],
    by_category: dict[str, list[StudentAttemptFixture]],
    max_cases: int,
    max_queries: int,
) -> list[tuple[Case, StudentAttemptFixture]]:
    """Pick the (case, student answer) queries to label, balanced and capped.

    Each case pairs with six archetypes, so "30 cases" is 180 queries and 900
    judgements at top-5 — six times the plan's budget of 150. A labelling job
    nobody finishes is worth less than a smaller one that gets done.

    Each case takes a **different** archetype, rotating by case index, so 30
    queries means 30 distinct cases spanning all six error types. Iterating
    archetype-major instead would put all 30 `correct` answers first and the
    cap would truncate before reaching any of the others — agreement measured
    on nothing but correct answers says little about how the proxy behaves on
    the answers that actually need feedback.

    Passes beyond the first offset the rotation again, so a larger
    `max_queries` keeps adding coverage rather than repeating itself.
    """
    eligible = [
        case for case in cases if by_category.get(case.ground_truth_finding.category.value)
    ][:max_cases]
    if not eligible:
        return []
    widest = max((len(v) for v in by_category.values()), default=0)
    queries: list[tuple[Case, StudentAttemptFixture]] = []
    seen: set[tuple[str, str]] = set()
    for rotation in range(widest):
        for index, case in enumerate(eligible):
            fixtures_here = by_category[case.ground_truth_finding.category.value]
            fixture = fixtures_here[(index + rotation) % len(fixtures_here)]
            key = (case.id, fixture.archetype)
            if key in seen:
                continue
            seen.add(key)
            queries.append((case, fixture))
            if len(queries) >= max_queries:
                return queries
    return queries[:max_queries]


def _pairs_for(
    queries: list[tuple[Case, StudentAttemptFixture]],
    retriever: IRetriever,
    corpus: Corpus,
    top_k: int,
) -> list[LabelledPair]:
    by_id = corpus.by_id
    pairs: list[LabelledPair] = []
    for case, fixture in queries:
        query = RetrievalQuery(
            free_text=fixture.text,
            confirmed_categories=case.confirmed_categories,
            named_categories=named_categories(fixture.text),
            case_id=case.id,
        )
        confirmed = set(case.confirmed_categories)
        for rank, passage in enumerate(
            retriever.retrieve(query=query, top_k=top_k), start=1
        ):
            chunk = by_id.get(passage.id)
            if chunk is None:
                continue
            # The proxy's verdict, computed exactly as the report computes it,
            # so agreement is measured against the real metric rather than a
            # reimplementation that could drift from it.
            proxy = bool(
                chunk.aspect not in _NON_TEACHING
                and confirmed.intersection(chunk.findings)
            )
            pairs.append(
                LabelledPair(
                    case_id=case.id,
                    category=case.ground_truth_finding.category.value,
                    archetype=fixture.archetype,
                    chunk_id=chunk.id,
                    chunk_aspect=chunk.aspect,
                    rank=rank,
                    proxy_relevant=proxy,
                )
            )
    return pairs


def _export(args: argparse.Namespace) -> int:
    settings = get_settings()
    corpus = load_corpus(settings.corpus_dir)
    cases = ManifestRepository(settings.manifest_path).list(split=args.split)
    fixtures = load_attempt_fixtures(FIXTURES_PATH)
    arm = resolve_feedback_arm(args.arm)
    retriever = build_retriever_for_arm(arm, corpus)

    by_category: dict[str, list[StudentAttemptFixture]] = {}
    for fixture in fixtures:
        by_category.setdefault(fixture.category, []).append(fixture)

    queries = _sample_queries(cases, by_category, args.cases, args.max_queries)
    pairs = _pairs_for(queries, retriever, corpus, args.top_k)

    path = Path(args.out) if args.out else DEFAULT_LABELS_PATH
    export_labelling_sheet(pairs, path)
    already = sum(1 for p in load_labels(path) if p.is_labelled)
    seen_cases = len({case.id for case, _ in queries})
    print(
        f"Exported {len(pairs)} pairs from {len(queries)} queries "
        f"across {seen_cases} cases to {path}"
    )
    print(f"Already labelled: {already}. Remaining: {len(pairs) - already}.")
    print(
        "\nFill in `label` on each pair (relevant | not_relevant | unsure), then:\n"
        "  python scripts/label_relevance.py validate"
    )
    return 0


def _validate(args: argparse.Namespace) -> int:
    path = Path(args.labels) if args.labels else DEFAULT_LABELS_PATH
    if not path.exists():
        raise SystemExit(f"No labels at {path}. Run `label_relevance.py export` first.")
    result = validate_proxy(load_labels(path))
    if not result.labelled:
        raise SystemExit(
            f"{path} has {result.total_pairs} pairs and none are labelled yet."
        )
    summary = render_agreement_summary(result)
    print("\n" + summary)
    out = path.with_name("relevance_agreement.md")
    out.write_text(summary + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {out}")
    return 0 if result.is_quotable else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="write the pairs to be labelled")
    export.add_argument("--cases", type=int, default=30, help="cases to draw from (default: 30)")
    export.add_argument(
        "--max-queries",
        type=int,
        default=30,
        help="cap on (case, answer) queries; 30 x top-5 = the plan's 150 judgements",
    )
    export.add_argument("--top-k", type=int, default=5, help="passages per query (default: 5)")
    export.add_argument("--split", default="test")
    export.add_argument("--arm", default="p3.0-bm25", help="arm to label (default: p3.0-bm25)")
    export.add_argument("--out", default=None)
    export.set_defaults(func=_export)

    validate = sub.add_parser("validate", help="score the proxy against the labels")
    validate.add_argument("--labels", default=None)
    validate.set_defaults(func=_validate)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
