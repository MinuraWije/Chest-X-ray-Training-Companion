"""The single `Feedback` -> API shape. Both routes call this; neither has its own.

There used to be two near-identical `_shape_feedback` functions, one in
`routes/attempts.py` and one in `routes/reviews.py`, and they had drifted:
the educator's copy hardcoded `differentialFindings: []` and put `key_signs`
in the `keyObservations` slot, so an educator reviewing an attempt saw
materially different feedback from the student who received it. On a marking
tool that is not cosmetic — the educator scores the student against text the
student never read. Phase 2 adds a third consumer (the eval's feedback path),
which is why this is one function now rather than three.

Both old copies also dropped `raw` entirely, which meant
`ground_truth_override` — the flag saying the displayed label is ours and not
the model's — never reached any client. It is surfaced here, through an
allowlist rather than by dumping `raw`: `raw` holds the whole parsed model
response, and shipping that wholesale would make every internal key part of
the public API by accident.
"""
from __future__ import annotations

from typing import Any, Final

from cxt.domain.entities.feedback import Feedback

__all__ = ["shape_feedback", "shape_passage", "PUBLIC_RAW_KEYS"]

# `raw` keys the clients may see, mapped to their camelCase DTO names.
# Provenance (NFR-4) and the ground-truth override flag; nothing that would
# let a student read the model's unfiltered response.
PUBLIC_RAW_KEYS: Final[dict[str, str]] = {
    "prompt_version": "promptVersion",
    "model": "model",
    "ground_truth_override": "groundTruthOverride",
    "retrieval_mode": "retrievalMode",
    "retrieval_k": "retrievalK",
    "corpus_version": "corpusVersion",
    "abstained": "abstained",
    "citations_stripped": "citationsStripped",
}


def shape_passage(passage: Any) -> dict[str, object]:
    """One retrieved passage as the frontend's `RetrievedPassage`.

    `id` and `url` are emitted even when empty so the frontend can key on `id`
    unconditionally; `source` is not unique (two chunks from one document
    collide) and was previously the de-facto key.
    """
    return {
        "id": passage.id,
        "source": passage.source,
        "url": passage.url,
        "text": passage.text,
        "score": float(passage.score),
    }


def _public_raw(feedback: Feedback) -> dict[str, object]:
    raw = feedback.raw if isinstance(feedback.raw, dict) else {}
    return {
        dto_key: raw[internal_key]
        for internal_key, dto_key in PUBLIC_RAW_KEYS.items()
        if internal_key in raw
    }


def shape_feedback(feedback: Feedback) -> dict[str, object]:
    """Re-shape the internal `Feedback` entity into the FE-facing `Feedback`."""
    label = feedback.likely_finding.value
    # `likely_finding` is the model's human-readable label ("Right apical
    # pneumothorax"); the canonical category is recorded separately by the
    # parser and may have been corrected by `_enforce_ground_truth`.
    raw = feedback.raw if isinstance(feedback.raw, dict) else {}
    category = raw.get("category") or label
    source = "ground_truth" if raw.get("ground_truth_override") else "model"
    key_observations = (
        feedback.key_observations.value
        if feedback.key_observations is not None
        else feedback.key_signs.value
    )
    return {
        "primaryFinding": {
            "category": category,
            "label": label,
            "confidence": float(feedback.confidence.value),
            "rationale": feedback.key_signs.value,
            "source": source,
        },
        "differentialFindings": [
            {
                "category": d.label,
                "label": d.label,
                "confidence": float(d.confidence.value),
                "rationale": "",
                "source": "model",
            }
            for d in feedback.differentials
        ],
        "keyObservations": key_observations,
        "teachingPoint": feedback.learning_point.value,
        "sources": [shape_passage(s) for s in feedback.sources],
        "raw": _public_raw(feedback),
    }
