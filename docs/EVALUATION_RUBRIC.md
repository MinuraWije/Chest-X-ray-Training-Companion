# Phase 1 Evaluation Rubric

> **Phase:** 1 of 3 (MedGemma only).
> **What this document is:** the metrics the foundational eval harness
> actually computes, with definitions, the per-finding rationale, and
> what's deliberately deferred to Phase 2 (RAG) and Phase 3 (fine-tune).
> **What it is not:** a clinical ground truth. NIH labels are NLP-mined
> radiology reports and are known to be noisy; treat the numbers as
> directional signals, not as a clinical reference.

## How to run

```powershell
cd backend
python scripts\dev.py data       # ensure data/processed/manifest.csv exists
python scripts\run_eval.py       # default: dry_run analyzer, 30 cases, test split
```

Outputs (in `backend/reports/` by default):

- `phase1_eval.json` — full machine-readable report
- `phase1_eval.md` — human-readable summary

CLI flags:

- `--max-cases N` — cap on cases (overrides `MAX_EVAL_CASES`)
- `--split test` — manifest split to evaluate (default: `test`; the held-out 15% from `build_manifest.py`)
- `--out PATH` — output JSON path (the `.md` summary sits alongside)
- `--no-markdown` — skip the human-readable summary
- `--runtime auto|dry_run|kaggle_tunnel` — override the analyzer runtime

## Phase 1 metrics (the only ones we report)

### 1. Structure validity

> `parse_succeeded / total_cases`

The fraction of cases whose raw model output parsed into a valid
`Feedback` entity via `OutputParser`. A failed parse means the JSON
extraction or schema validation rejected the output (empty body,
unrecognised category, out-of-range confidence, etc.).

**What it tells you:** "Is the model returning *anything* we can
use, or are we throwing most outputs away?" Low structure validity
points to prompt drift (model is adding prose) or to model confusion
(model is returning an empty or off-schema response).

**What it does not tell you:** whether the *content* is correct. A
parser-bypass attack would have 100% structure validity and 0%
finding recognition.

### 2. Finding recognition

> `correct / total_cases`

Where `correct = (parsed && predicted_category == ground_truth_category)`.
The ground truth comes from the manifest's `ground_truth_finding.category`
column, which is mapped from the NIH label using the table in
`docs/DATA_NIH.md` (e.g. NIH `Pneumothorax` → `pneumothorax`).

**What it tells you:** the model's ability to pick the *correct* one
of the seven categories. A flat 60% is reasonable for a small
vision-language model on noisy labels.

**What it does not tell you:** whether the rationale, key signs, or
teaching point is any good. Those are qualitative.

### 3. Per-finding recognition

> `correct(category) / total(category)` for each `category` in
> `{pneumothorax, pleural_effusion, cardiomegaly, consolidation,
> nodule_or_mass, fracture, no_finding}`.

**Why this exists:** NIH labels are noisy and the seven categories are
not equally well-represented in the data. A flat accuracy masks
"the model nails pneumothorax but never sees a fracture". The
per-finding breakdown is the table you read when planning Phase 3
(fine-tuning needs more of the underperforming categories).

**Reading the table:**
- If a category has `total = 0`, it is missing from the eval split.
  Either re-build the manifest with `--n-per-finding 5` for that
  category or seed the corpus more aggressively in Step 3.
- If a category has `total > 0` and `accuracy < 0.4`, the model is
  not learning the visual signature; flag for the prompt-experiments
  notebook (`backend/notebooks/prompt_experiments.py`).

### 4. Latency

> `latency_mean_ms`, `latency_p50_ms`, `latency_p95_ms`

Wall-clock around `analyzer.analyze(...)`. We measure the *whole*
analyzer call (model call + parser). The dry-run analyzer reports
~0–5 ms; the Kaggle-tunnel analyzer is dominated by the round-trip
plus MedGemma inference (≈ 2–8 s on a T4 with 4-bit quantisation).

**Why p50 and p95 (not just mean):** the first request of a session
is always slower (model warm-up, ngrok handshake). p95 surfaces the
*long tail* of slow requests, which is what the UI's worst-case
spinner will look like.

**Targets for Phase 1:**
- `latency_p95_ms` should be < 15 000 ms on the Kaggle runtime.
- The dry-run analyzer is for dev/test only; do not gate UI work on
  its latency.

### 5. Token usage

> `total_prompt_tokens`, `total_output_tokens` summed across all cases.

**Phase 1 truth:** the dry-run analyzer reports token *estimates* based
on whitespace splits. The Kaggle-tunnel analyzer reports the model's
*own* `prompt_tokens` / `output_tokens` from the `transformers` call
(this is what the Kaggle runtime returns and what the backend logs).
Don't compare the two numerically — they are not on the same scale.
Use the Kaggle-tunnel numbers for cost / quota planning; use the
dry-run numbers only to spot prompt explosions during local dev.

**Targets for Phase 1:**
- The system prompt is ~250 tokens; the user prompt is ~30 tokens.
- Each output should be ~200–400 tokens. Above 500 tokens per case
  means the model is being verbose (often because the prompt asked
  for differentials or extra fields).
- A full 30-case eval should fit comfortably inside Kaggle's free
  tier quota (~ 10 000 total tokens).

## Sample report (what the JSON looks like)

```json
{
  "runtime": "kaggle_tunnel",
  "total_cases": 30,
  "parse_succeeded": 28,
  "parse_failed": 2,
  "structure_validity": 0.933,
  "finding_recognition": 0.567,
  "per_finding_recognition": {
    "no_finding":        { "total": 5.0, "correct": 4.0, "accuracy": 0.8 },
    "pneumothorax":      { "total": 5.0, "correct": 3.0, "accuracy": 0.6 },
    "pleural_effusion":  { "total": 4.0, "correct": 2.0, "accuracy": 0.5 },
    "cardiomegaly":      { "total": 4.0, "correct": 3.0, "accuracy": 0.75 },
    "consolidation":     { "total": 4.0, "correct": 1.0, "accuracy": 0.25 },
    "nodule_or_mass":    { "total": 4.0, "correct": 2.0, "accuracy": 0.5 },
    "fracture":          { "total": 4.0, "correct": 2.0, "accuracy": 0.5 }
  },
  "latency_mean_ms": 4210.3,
  "latency_p50_ms": 3870.0,
  "latency_p95_ms": 8120.0,
  "total_prompt_tokens": 8240,
  "total_output_tokens": 9615,
  "model": "google/medgemma-1.5-4b-it",
  "prompt_version": "e2.0+a1b2c3d4e5f6",
  "split": "test",
  "sampling": "stratified",
  "samples": [ ... ]
}
```

## Provenance and sampling

Four fields exist so a report can be read months later without guessing
what produced it:

- **`model`** — observed, not configured. It is the identifier the
  inference server reported back per case, so the report states what
  actually answered rather than what was meant to be called.
- **`prompt_version`** — the eval prompt set, as `<version>+<fingerprint>`.
  The fingerprint is a hash of the prompt text: an edit that forgets to
  bump the version still changes the fingerprint, so two runs on
  different wording can never be silently compared. Select a set with
  `--prompt-version`.
- **`split`** and **`sampling`** — which manifest split, and how cases
  were drawn from it.

Sampling defaults to **stratified**: cases are round-robined across the
ground-truth categories present in the split, deterministically (manifest
order within each category), so two runs are comparable and every
category contributes a similar number of cases. This matters because the
curated NIH subset is not balanced — taking the first N manifest rows
produced a 30-case run holding 11 cardiomegaly, 3 consolidation and no
negatives at all, where "consolidation: 0.0%" was really 0/3 and carried
no signal. `--no-stratify` restores that first-N behaviour for
reproducing an older run.

## What's deliberately NOT in Phase 1 (deferred)

| Metric | When it lands |
|---|---|
| Retrieval relevance (precision@5, MRR) | **Phase 2** (with RAG) |
| Evidence grounding (does the rationale cite the corpus) | **Phase 2** |
| Learning usefulness (does the feedback help a learner) | qualitative only in Phase 1; rubric form in Phase 3 |
| Calibration (Brier score, reliability diagram) | **Phase 3** (after fine-tune) |
| Inter-rater agreement (Cohen's κ) | not in scope — single-model baseline |
| Latency budget under contention | **Phase 2** (load test with the RAG retriever in the loop) |
| Per-dataset eval (NIH vs MIMIC) | **Phase 1.5** (MIMIC swap) |

## How to interpret a low score

Worked example: "`consolidation` accuracy 25% across 4 cases".

1. **Look at the raw outputs** — they are in the JSON report under
   `samples[*].raw_text` (only if you kept the raw field; the default
   `OutputParser` stores `raw_text` separately). Was the model saying
   "consolidation" in prose but emitting the wrong JSON? Or is it
   truly confused about the image?
2. **Run the prompt-experiments notebook** —
   `backend/notebooks/prompt_experiments.py`. Try the `concise`
   variation (category + confidence only) and the `verbose` variation
   (full feedback schema). One of them usually surfaces the bug.
3. **Check the manifest label** — NIH consolidation is
   `Consolidation + Infiltration + Pneumonia → consolidation`. If
   you see the model saying "infiltrate" or "opacity", the parser
   is right to call it wrong; the label *or* the model is off, not
   the parser.
4. **Surface to the dev plan** — Phase 1 is intentionally narrow; if
   `consolidation` is consistently poor, Phase 3 should add a
   consolidation-specific fine-tune pass.

## Files

- `backend/src/cxt/infrastructure/eval/__init__.py` — `run_evaluation`,
  `write_json_report`, `write_markdown_summary`, the `EVAL_SYSTEM_PROMPT`
  and `EVAL_USER_PROMPT` constants.
- `backend/scripts/run_eval.py` — CLI entrypoint.
- `backend/notebooks/prompt_experiments.py` — `evaluate_variation(...)`
  for one-off prompt A/B tests.
- `backend/src/cxt/infrastructure/ml/output_parser.py` — what counts
  as a "valid" parse.
- `backend/src/cxt/infrastructure/data/manifest_repository.py` — where
  the ground-truth categories come from.
- `docs/DATA_NIH.md` — the NIH → six-findings mapping.
