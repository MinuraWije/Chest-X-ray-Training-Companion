"""Manual relevance labels, and validating the automatic proxy against them.

Precision@3 in the Phase 2 report is computed from **tag overlap**: a retrieved
chunk counts as relevant if one of its `findings` tags matches a category the
case confirms. That is cheap, deterministic and obviously incomplete — a chunk
tagged `pleural_effusion` on an effusion case scores as relevant even when it
is about a sign the film does not show, and a chunk about the exact confusion
the student fell into scores as irrelevant if its tags happen not to match.

So the number needs a second opinion. This module supports the workflow the
plan called high value for half a day of effort:

1. `export_labelling_sheet` writes every (case, student answer, retrieved
   chunk) pair to a JSON file with `label: null`.
2. A human fills in `relevant` / `not_relevant` / `unsure`, reading the chunk
   against what the student actually wrote.
3. `validate_proxy` reports how often the automatic tag-overlap judgement
   agrees with the human one.

"My cheap automatic metric agrees with my manual labels on 82% of pairs" is a
far stronger claim than either number alone, and it is honest about the
instrument in a way a bare precision figure is not.

**Cohen's kappa is reported alongside raw agreement.** Raw agreement is
inflated whenever one label dominates, and it will: most retrieved chunks on a
5-finding corpus are topically related, so a proxy that answered "relevant" to
everything would score well above chance on agreement alone. Kappa corrects for
that and is the number to quote if the two disagree.

The labels are data the human produced, so they are committed; nothing here
generates them.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from cxt.shared.errors import ValidationError

__all__ = [
    "RelevanceLabel",
    "LabelledPair",
    "ProxyAgreement",
    "VALID_LABELS",
    "export_labelling_sheet",
    "load_labels",
    "validate_proxy",
    "render_agreement_summary",
]

RelevanceLabel = str

#: `unsure` is a first-class option, not a cop-out. Forcing a binary judgement
#: on a genuinely ambiguous pair manufactures agreement or disagreement that
#: the labeller does not actually hold. Unsure pairs are excluded from the
#: agreement computation and their count is reported.
VALID_LABELS: Final[frozenset[str]] = frozenset({"relevant", "not_relevant", "unsure"})

#: Below this many labelled pairs, an agreement figure is not worth quoting.
#: The plan budgeted ~30 cases x top-5 = 150 judgements.
MIN_PAIRS_FOR_AGREEMENT: Final[int] = 30


@dataclass(frozen=True, slots=True)
class LabelledPair:
    """One (query, chunk) pair, with both the human and the automatic verdict."""

    case_id: str
    category: str
    archetype: str
    chunk_id: str
    chunk_aspect: str
    rank: int
    proxy_relevant: bool
    label: RelevanceLabel | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.label is not None and self.label not in VALID_LABELS:
            raise ValidationError(
                f"Label {self.label!r} for {self.chunk_id!r} is not one of "
                f"{sorted(VALID_LABELS)}."
            )

    @property
    def is_labelled(self) -> bool:
        return self.label is not None

    @property
    def is_decisive(self) -> bool:
        """Labelled, and not `unsure`."""
        return self.label in {"relevant", "not_relevant"}

    @property
    def human_relevant(self) -> bool:
        return self.label == "relevant"

    @property
    def agrees(self) -> bool:
        return self.is_decisive and (self.human_relevant == self.proxy_relevant)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "archetype": self.archetype,
            "chunk_id": self.chunk_id,
            "chunk_aspect": self.chunk_aspect,
            "rank": self.rank,
            "proxy_relevant": self.proxy_relevant,
            "label": self.label,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LabelledPair:
        return cls(
            case_id=str(payload["case_id"]),
            category=str(payload.get("category", "")),
            archetype=str(payload.get("archetype", "")),
            chunk_id=str(payload["chunk_id"]),
            chunk_aspect=str(payload.get("chunk_aspect", "")),
            rank=int(payload.get("rank", 0)),
            proxy_relevant=bool(payload.get("proxy_relevant", False)),
            label=payload.get("label"),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True, slots=True)
class ProxyAgreement:
    """How well tag overlap tracks human judgement."""

    total_pairs: int
    labelled: int
    unsure: int
    decisive: int
    agreements: int
    # Confusion counts, proxy vs human.
    both_relevant: int = 0
    both_irrelevant: int = 0
    proxy_only: int = 0
    human_only: int = 0
    by_aspect: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def agreement(self) -> float:
        """Raw agreement. Inflated when one label dominates — quote kappa too."""
        return self.agreements / self.decisive if self.decisive else 0.0

    @property
    def cohens_kappa(self) -> float:
        """Agreement corrected for what chance alone would produce.

        Returns 0.0 when either rater used only one label, because kappa is
        undefined there — and that situation is itself the finding: a proxy
        that says "relevant" to everything has no discriminating power however
        high its raw agreement.
        """
        n = self.decisive
        if not n:
            return 0.0
        observed = self.agreement
        proxy_rel = (self.both_relevant + self.proxy_only) / n
        human_rel = (self.both_relevant + self.human_only) / n
        expected = proxy_rel * human_rel + (1 - proxy_rel) * (1 - human_rel)
        if expected >= 1.0:
            return 0.0
        return (observed - expected) / (1 - expected)

    @property
    def is_quotable(self) -> bool:
        return self.decisive >= MIN_PAIRS_FOR_AGREEMENT

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pairs": self.total_pairs,
            "labelled": self.labelled,
            "unsure": self.unsure,
            "decisive": self.decisive,
            "agreement": round(self.agreement, 4),
            "cohens_kappa": round(self.cohens_kappa, 4),
            "quotable": self.is_quotable,
            "confusion": {
                "both_relevant": self.both_relevant,
                "both_irrelevant": self.both_irrelevant,
                "proxy_relevant_human_not": self.proxy_only,
                "human_relevant_proxy_not": self.human_only,
            },
            "by_aspect": {k: {"agree": v[0], "of": v[1]} for k, v in self.by_aspect.items()},
        }


def export_labelling_sheet(pairs: list[LabelledPair], path: Path) -> Path:
    """Write the pairs to label, preserving any labels already present.

    Re-running after retrieval changes must not silently discard work: pairs
    keyed (case_id, archetype, chunk_id) that already carry a label keep it,
    and only genuinely new pairs arrive with `label: null`.
    """
    existing: dict[tuple[str, str, str], LabelledPair] = {}
    if path.exists():
        for old in load_labels(path):
            existing[(old.case_id, old.archetype, old.chunk_id)] = old

    merged: list[LabelledPair] = []
    for pair in pairs:
        previous = existing.get((pair.case_id, pair.archetype, pair.chunk_id))
        if previous is not None and previous.is_labelled:
            # Keep the human's verdict, refresh the proxy's — the point of the
            # exercise is to see whether the proxy moved toward the human.
            merged.append(
                LabelledPair(
                    **{**pair.to_dict(), "label": previous.label, "note": previous.note}
                )
            )
        else:
            merged.append(pair)

    payload = {
        "_readme": [
            "Manual relevance labels for the Phase 2 retrieval evaluation.",
            "",
            "Set `label` on each pair to one of: relevant, not_relevant, unsure.",
            "Judge the chunk against what the STUDENT WROTE and what the case",
            "confirms — not against whether it is about the right finding in the",
            "abstract. A chunk on the correct topic that would not help this",
            "student with this answer is `not_relevant`.",
            "",
            "`unsure` is a real option. Forcing a binary on a genuinely ambiguous",
            "pair manufactures agreement the labeller does not hold; unsure pairs",
            "are excluded from the agreement figure and counted separately.",
            "",
            "`proxy_relevant` is what the automatic tag-overlap metric decided.",
            "Do not read it before labelling — it is recorded so agreement can be",
            "computed afterwards, not as a hint.",
        ],
        "pairs": [p.to_dict() for p in merged],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_labels(path: Path) -> tuple[LabelledPair, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "pairs" not in payload:
        raise ValidationError(f"{path}: expected an object with a 'pairs' list.")
    return tuple(LabelledPair.from_dict(row) for row in payload["pairs"])


def validate_proxy(pairs: tuple[LabelledPair, ...]) -> ProxyAgreement:
    """Compare the automatic tag-overlap verdict against the human labels."""
    labelled = [p for p in pairs if p.is_labelled]
    decisive = [p for p in labelled if p.is_decisive]

    both_relevant = sum(1 for p in decisive if p.human_relevant and p.proxy_relevant)
    both_irrelevant = sum(
        1 for p in decisive if not p.human_relevant and not p.proxy_relevant
    )
    proxy_only = sum(1 for p in decisive if p.proxy_relevant and not p.human_relevant)
    human_only = sum(1 for p in decisive if p.human_relevant and not p.proxy_relevant)

    # Per aspect, because the proxy is expected to fail differently on
    # `management` chunks (topical, unhelpful) than on `radiographic_sign`.
    agree_counts: Counter[str] = Counter()
    total_counts: Counter[str] = Counter()
    for pair in decisive:
        total_counts[pair.chunk_aspect] += 1
        if pair.agrees:
            agree_counts[pair.chunk_aspect] += 1

    return ProxyAgreement(
        total_pairs=len(pairs),
        labelled=len(labelled),
        unsure=len(labelled) - len(decisive),
        decisive=len(decisive),
        agreements=both_relevant + both_irrelevant,
        both_relevant=both_relevant,
        both_irrelevant=both_irrelevant,
        proxy_only=proxy_only,
        human_only=human_only,
        by_aspect={
            aspect: (agree_counts[aspect], total_counts[aspect]) for aspect in total_counts
        },
    )


def render_agreement_summary(result: ProxyAgreement) -> str:
    lines = [
        "# Relevance proxy validation",
        "",
        f"- Pairs exported: **{result.total_pairs}**",
        f"- Labelled: **{result.labelled}** ({result.unsure} unsure, excluded)",
        f"- Raw agreement: **{result.agreement:.0%}** of {result.decisive} decisive pairs",
        f"- Cohen's kappa: **{result.cohens_kappa:.2f}**",
        "",
    ]
    if not result.is_quotable:
        lines.extend(
            [
                f"> **Not quotable.** {result.decisive} decisive pairs is below the "
                f"{MIN_PAIRS_FOR_AGREEMENT} needed for the figure to mean anything. "
                "Label more pairs before citing this in the report.",
                "",
            ]
        )
    lines.extend(
        [
            "## Where they disagree",
            "",
            "| | Human: relevant | Human: not relevant |",
            "|---|---|---|",
            f"| **Proxy: relevant** | {result.both_relevant} | {result.proxy_only} |",
            f"| **Proxy: not relevant** | {result.human_only} | {result.both_irrelevant} |",
            "",
            f"- **{result.proxy_only}** pairs the tag overlap called relevant and a "
            "human did not — chunks on the right topic that would not help this "
            "student. This is the proxy's expected failure mode.",
            f"- **{result.human_only}** pairs a human called relevant and the tags "
            "missed — usually a chunk addressing the student's confusion whose "
            "`findings` tags do not name the confirmed category.",
            "",
            "## By aspect",
            "",
            "| Aspect | Agreement |",
            "|---|---|",
        ]
    )
    for aspect, (agree, total) in sorted(result.by_aspect.items()):
        share = agree / total if total else 0.0
        lines.append(f"| `{aspect}` | {share:.0%} ({agree}/{total}) |")
    lines.extend(
        [
            "",
            "Raw agreement is inflated when one label dominates, which it does here: "
            "most retrieved chunks on a five-finding corpus are topically related, so "
            "a proxy answering \"relevant\" to everything would still score highly. "
            "Quote kappa if the two figures disagree.",
            "",
        ]
    )
    return "\n".join(lines)
