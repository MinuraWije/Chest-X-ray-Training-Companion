# Enable real MedGemma inference (Phase 1 production path)

## Step 0 — Commit this plan to the repo

`docs/plans/` does not exist yet. Create it and write this document to
`docs/plans/2026-09-05-medgemma-support.md` so the plan lives alongside
`DEVELOPMENT_PLAN.md` and can be cited from the report's implementation chapter.
This is the first action taken on approval, before any code changes.

## Context

Every piece of the MedGemma path is written — `KaggleTunnelMedGemmaClient`, `MedGemmaAnalyzer`, `OutputParser`, the prompt templates, the Kaggle notebook server, and the `MEDGEMMA_RUNTIME=kaggle_tunnel` switch in `dependencies.py:127-135`. None of it has ever produced a real model response. `backend/reports/phase1_eval.md` reports `Runtime: dry_run`, `latency 0ms`, and a 23.3% finding-recognition figure that is the **stub analyzer's** number, not MedGemma's.

The reason is a single unresolved seam. `ManifestRepository._to_case` builds `image_url = "/images/00000013_023.png"` (`manifest_repository.py:57-58`) — a browser-relative path, correct for the SPA because `app.py:52-57` mounts `StaticFiles` there. `SubmitAttemptUseCase` then passes that string verbatim to the analyzer (`submit_attempt.py:106`), the client forwards it unchanged (`kaggle_tunnel_client.py:66`), and the notebook hands it to `transformers` as an image location (`medgemma_server.py:138`). Inside the Kaggle container that path does not exist, so **every non-dry-run submission fails**. The dry-run analyzer masks this completely — it only hashes the string to pick a canned payload (`dry_run_analyzer.py:125-126`), so it never notices the path is bogus.

Fixing transport alone is not enough to get a usable result. Four further defects only surface once a real model answers:

- **Cold start.** The notebook loads the 4B checkpoint lazily, inside the first `/generate` call (`medgemma_server.py:98-103, 126-129`). Download + 4-bit quantize takes minutes; the client's timeout is 60s with 3 hardcoded retries, each blocking on `PIPELINE_LOCK`. The first real submission times out three times and gives up.
- **The student's essay lands in the system prompt.** `PromptBuilder.build` interpolates `{student_free_text}` into the *system* text (`prompts.py:35-38`, `system.txt:29`), which the server caps at 4000 chars (`medgemma_server.py:72`). `POST /attempts` never bounds `freeText` (`routes/attempts.py:133-134`), so a long answer produces a 422 that the client reports as an upstream failure. Separately, `str.format` means a student typing a literal `{` raises `KeyError` before the model is ever called.
- **No repair path for imperfect JSON.** `DEVELOPMENT_PLAN.md:402` specifies "one retry with a corrective prompt"; it was never built. Any parse failure is a hard 503 (`output_parser.py:59-70`). A 4-bit 4B model routinely emits prose-wrapped or token-truncated JSON, so structure validity will read far worse than the model deserves — and `MEDGEMMA_MAX_NEW_TOKENS=300` is too small for the 4-field schema, guaranteeing truncation.
- **The model's richest output is discarded.** `output_parser.py:107-116` builds `key_observations` and then `del`s it; `routes/attempts.py:106` hardcodes `differentialFindings: []` and line 107 sets `keyObservations` to a copy of `key_signs`. Real inference would render in the UI identically to dry-run.

**Outcome:** a student submission reaches MedGemma on Kaggle and returns genuine structured feedback; `scripts/run_eval.py --runtime kaggle_tunnel` produces a `phase1_eval.json` carrying real recognition, latency and token figures. That artefact is what unblocks the missing Results chapter of the report and the two outstanding 50%-milestone checkboxes (`DEVELOPMENT_PLAN.md:512, 515`).

**Decisions taken:** images travel as **base64 bytes in the request** (self-contained, works from any machine, exposes nothing of the dev machine publicly, and keeps the privacy narrative in `docs/KAGGLE_RUNTIME.md:94-107` honest). Scope includes the corrective-retry path, restoring the discarded output, and a real eval run.

---

## Phase 1 — Image transport (the blocker)

### 1a. Widen the port to carry an image reference, not a path

The port only accepts a string (`domain/ports/medgemma_client.py:25`), so bytes cannot be expressed at all today. Add a value object beside the existing ones in `backend/src/cxt/domain/value_objects/`:

```python
# cxr_image_ref.py
@dataclass(frozen=True, slots=True)
class CxrImageRef:
    uri: str                      # canonical id for logging/audit: "/images/00000013_023.png"
    data_b64: str | None = None   # inline bytes when the runtime cannot resolve uri
    mime: str = "image/png"
```

Follow the `__post_init__` + `ValidationError` convention used by `confidence.py` / `rationale.py`. Keeping `uri` alongside the bytes preserves NFR-6 logging (log the case id, never the bytes) and means the eval harness can still group by case.

Change both port signatures to take `image: CxrImageRef` instead of `image_url_or_path: str`:
- `domain/ports/medgemma_client.py:25` — `IMedGemmaClient.generate`
- `domain/ports/image_analyzer.py:23` — `IImageAnalyzer.analyze`

Call sites to update: `submit_attempt.py:105-109`, `medgemma_analyzer.py:29-35`, `kaggle_tunnel_client.py:58-70`, `dry_run_analyzer.py` (hash `image.uri` so canned output stays stable), `infrastructure/eval/__init__.py:208`, `notebooks/prompt_experiments.py:~105`.

### 1b. Resolve `/images/...` to bytes

New `backend/src/cxt/infrastructure/ml/image_loader.py`:

```python
class LocalImageLoader:
    def __init__(self, images_dir: Path, *, max_bytes: int = 8 * 1024 * 1024) -> None: ...
    def load(self, image_url: str) -> CxrImageRef: ...
```

Strips the `/images/` prefix, resolves against `settings.images_dir`, and **must** assert the resolved path stays inside `images_dir` (reject `..` traversal) and is under `max_bytes`. Returns a `CxrImageRef` with `uri` set to the original `/images/...` string and `data_b64` populated. Raise `ValidationError` on a missing or oversized file so the 503 message is legible.

Wire it in `dependencies.py` next to `get_case_repository` (`:116-121`) and inject into `SubmitAttemptUseCase`; the eval harness constructs its own in `run_eval.py`. Only the `kaggle_tunnel` runtime needs bytes — for `dry_run`, pass a `CxrImageRef` with `data_b64=None` to avoid pointless disk I/O in tests.

### 1c. Accept bytes on the notebook side

`kaggle_runtime/medgemma_server.py` — extend `GenerateRequest` (`:69-73`):

```python
image_url_or_path: str | None = Field(default=None, max_length=2000)
image_base64: str | None = None
image_mime: str = "image/png"
```

Add a validator requiring exactly one of the two image fields. In `generate` (`:124`), when `image_base64` is set, decode and **write to a `NamedTemporaryFile`, then pass that path** in the existing `{"type": "image", "url": <path>}` slot (`:138`), deleting it in a `finally`. Going via a temp file rather than handing the pipeline a PIL object keeps the change to one line of the message construction and avoids depending on version-specific `transformers` content-item handling. Keep the `image_url_or_path` branch working — it is what the `docs/KAGGLE_RUNTIME.md:133-146` smoke snippet uses.

Log `image_mime` and byte length only; never the payload (the server's existing logging discipline, `:144-149`).

### 1d. Fix cold start

In `start_in_background` (`medgemma_server.py:179-198`), call `_ensure_pipeline(...)` **before** `ngrok.connect(...)`. The model is then resident by the time the URL is printed and pasted into `.env`, which removes the failure mode entirely without any backend timeout gymnastics. Add a `POST /warmup` endpoint for re-warming, and raise the `MEDGEMMA_TIMEOUT_SECONDS` default from 60 to 120 in `settings.py:62` and `.env.example:47`.

---

## Phase 2 — Make real output survive the round trip

### 2a. Move student text into the user turn

Edit `kaggle_runtime/prompts/system.txt` to drop the trailing `The student wrote: "{student_free_text}"` line, and add it to `user.txt`. This is better prompt modelling anyway — the student's answer is user-turn content — and it moves the unbounded field from the 4000-char `system` cap to the 8000-char `prompt` cap.

In `prompts.py:32-40`, replace `str.format` with `string.Template.safe_substitute` (or explicit `.replace()`) so a literal `{` or `$` in student text cannot raise. Add a `PROMPT_VERSION` constant plus a template hash, and record both in `Feedback.raw` — `NFR-4` in the report (Table 3.8) promises exactly this and it is not currently done.

Enforce `freeText` length at the API (`routes/attempts.py:133-134`) to match the frontend's existing Zod 20–2000 range, returning a 422 with a field message rather than letting Kaggle reject it.

### 2b. Repair and retry malformed JSON

Two layers.

**Parser repair** — extend `_extract_json` (`output_parser.py:42-70`). Today it only handles fences and surrounding prose despite the docstring at `:3-5` claiming more. Add: trailing-comma strip, smart-quote normalisation, and **truncation repair** (if the object ends mid-structure, balance the open braces/brackets and retry). Truncation is the single most likely failure at 4-bit; this alone will lift structure validity materially. Also raise `MEDGEMMA_MAX_NEW_TOKENS` from 300 → 512 (`settings.py:61`, `.env.example:46`) — 300 does not fit the 4-field schema.

**Corrective re-prompt** — new `backend/src/cxt/infrastructure/ml/resilient_analyzer.py`:

```python
class RetryingFeedbackParser:
    def __init__(self, analyzer: IImageAnalyzer, parser: OutputParser, *, max_repair_attempts: int = 1) -> None: ...
    def analyze_and_parse(self, *, attempt_id: str, image: CxrImageRef, prompt: str, system: str) -> tuple[Feedback, ModelOutput]: ...
```

On a parse failure it re-calls once with a corrective suffix appended to the user prompt (quoting the offending output and restating the required keys), then gives up with the original `UpstreamUnavailableError`. Put the retry here rather than in `OutputParser` so the parser stays pure and the golden tests in `tests/backend/golden/` keep working unchanged. Use it from both `SubmitAttemptUseCase._generate_feedback` (`submit_attempt.py:101-114`) and `run_evaluation` (`eval/__init__.py:191-235`) so the eval numbers reflect the same behaviour students get.

### 2c. Stop throwing away differentials and observations

- `domain/entities/feedback.py:48-56` — add `key_observations: Rationale` and `differentials: tuple[DifferentialFinding, ...]` (a small frozen dataclass beside `RetrievedPassage`). Give both defaults so previously-stored `feedback_json` rows still deserialize.
- `output_parser.py:107-116` — delete the `del key_observations_rationale` line and populate the new field; populate `differentials` from `differential_findings` (the `_format_missed_clues` prose summary at `:183-199` can stay as `missed_clues`).
- `routes/attempts.py:106-107` — map the real values instead of `[]` and the duplicated `key_signs`.
- Update `tests/backend/golden/feedback_dto.json` to match.
- Check `frontend/src/components/FeedbackSection.tsx:23` — it already destructures `differentialFindings` and `keyObservations`, so this may need no frontend change beyond verifying it renders a non-empty differentials list sensibly.

---

## Phase 3 — Real evaluation run

- `backend/scripts/dev.py:242-243` — `cmd_eval` currently invokes `run_eval.py` with **zero arguments**, so the runtime flag cannot be reached through it. Pass `argv` through.
- `medgemma_server.py:163` — `output_tokens` is `len(assistant.split())`, a word count, not tokens. Use the tokenizer. "Token usage" is a reported metric in the thesis (`EVALUATION_RUBRIC.md`), so it needs to be real.
- `run_eval.py` — record the `model` field the server already returns (`medgemma_server.py:169`, currently ignored by the client at `kaggle_tunnel_client.py:88-92`) and the prompt version into `EvalReport`, so the report can state exactly which model and prompt produced the numbers.
- Run against the live tunnel and commit the resulting `reports/phase1_eval.{json,md}`.

---

## Phase 4 — Correctness cleanups found along the way

Small, all in files already being touched:

- `kaggle_tunnel_client.py:96` — `stop_after_attempt(3)` is hardcoded; `settings.medgemma_max_retries` is inert, used only in an error string at `:82`. Swap the decorator for a `tenacity.Retrying(...)` object built from settings.
- `dependencies.py:135` builds a fresh `KaggleTunnelMedGemmaClient` — and therefore a fresh `httpx.Client` — **per request**, and nothing ever calls `close()` (`kaggle_tunnel_client.py:124`). Cache one client for the process (or manage it in the app lifespan).
- Add `ngrok-skip-browser-warning: true` to the request headers (`:102`). Its absence is the classic cause of the "tunnel returned non-JSON" error at `:117-119`.
- `MEDGEMMA_DRY_RUN` is a **phantom flag** — documented at `docs/KAGGLE_RUNTIME.md:39,111,129`, `kaggle_runtime/README.md:55,60`, and in the error message at `dependencies.py:132`, but no such field exists on `Settings` and `extra="ignore"` (`settings.py:41`) swallows it. Correct all five references to `MEDGEMMA_RUNTIME`.
- `medgemma_analyzer.py:27` — `prompt_builder` is a dead field whose `default_factory` does disk I/O for prompts it never uses, and will raise `FileNotFoundError` if `kaggle_runtime/prompts/` is absent. Remove it.
- `kaggle_tunnel_client.py:48-56` — `ping()` exists but is called nowhere. Surface it on `GET /api/v1/healthz`, which today only reports `tunnel_configured: bool(...)` (`app.py:83`).
- `medgemma_server.py:36-38` — the docstring's `%cd /kaggle/cxt/kaggle_runtime` then `from kaggle_runtime.medgemma_server import ...` cannot work from inside the package dir. `kaggle_runtime/README.md:25` has it right.
- `dev.py:263` — `cmd_all` calls `cmd_data` with an empty `Namespace`, so `args.source` raises `AttributeError`. `all` is broken.

---

## Tests

Match the existing conventions exactly: `from __future__ import annotations` first, full annotations (repo is `mypy --strict`), hand-rolled `_Stub`-prefixed doubles — **`unittest.mock` is used nowhere in this repo** — and class-based grouping (`TestOutputParserErrors`, `TestRunEvaluation`) for anything over ~4 cases.

- **`FakeMedGemmaClient`** — there is currently **no** double at the `IMedGemmaClient` level anywhere, and `MedGemmaAnalyzer` has **zero tests**. Add both. `tests/conftest.py` is 11 lines of `sys.path` bootstrapping with no fixtures, so this is the first shared double; while adding it, consolidate the duplicated `_StubAnalyzer` (`test_eval_harness.py:82` and `test_prompt_experiments.py:33`) and `_StubCaseRepo` (`test_eval_harness.py:61` and `test_submit_attempt.py:35`).
- `LocalImageLoader` — path traversal rejected, missing file, oversize file, correct base64 round-trip.
- `KaggleTunnelMedGemmaClient` — assert the POST body carries `image_base64`/`image_mime`; reuse the existing `_StubTransport` (`test_kaggle_tunnel_client.py:18`).
- `RetryingFeedbackParser` — repairs on second call; gives up after `max_repair_attempts`; does not re-call on a clean first parse.
- Parser repair — truncated JSON, trailing comma, smart quotes, prose-wrapped. Add a truncated-output golden pair alongside the existing `medgemma_response.json` / `feedback_dto.json`.
- **Live smoke test** — `@pytest.mark.slow`, `skipif` when `MEDGEMMA_TUNNEL_URL` is unset. The marker is registered at `pyproject.toml:45-47` with `--strict-markers`, but is used nowhere yet; `DEVELOPMENT_PLAN.md:478` asks for exactly this test.

Note `tests/backend/unit/infrastructure/eval/` and `.../notebooks/` are missing `__init__.py` while every sibling has one — add them for consistency.

---

## Docs

- `docs/KAGGLE_RUNTIME.md` — new request contract (base64 fields, exactly-one-of), eager warm-up, corrected `MEDGEMMA_RUNTIME` guidance, updated privacy paragraph (`:94-107`) now that bytes travel over the tunnel rather than a URL being dereferenced.
- `kaggle_runtime/README.md` — same contract change, fix the `%cd` instruction.
- `backend/.env.example` — new/changed defaults. Also worth noting: `APP_ENV`, `APP_HOST`, `APP_PORT`, `APP_LOG_LEVEL`, `APP_CORS_ORIGINS`, `ADMIN_EMAIL`, `ADMIN_PASSWORD` (lines 8-14, 27-28) map to **no `Settings` field** and are silently ignored — the real fields are `environment` and `cors_allow_origins`.

---

## Verification

**Offline (must stay green throughout):**
```powershell
cd backend
python scripts\dev.py test          # 257 existing + new; nothing regresses
python scripts\dev.py lint          # ruff + mypy --strict
cd ..\frontend && npm run build     # tsc -b, catches the Feedback DTO shape change
```

**Dry-run end-to-end** — confirms the port widening did not break the default path:
```powershell
python scripts\dev.py real-data ; python scripts\dev.py seed ; python scripts\dev.py run
# register a student at 127.0.0.1:5173/register, open a case, submit — feedback still renders
```

**Live tunnel:**
1. Run the Kaggle notebook; confirm the model loads *before* the ngrok URL prints.
2. `GET https://<url>/healthz` → `pipeline_loaded: "True"`.
3. Paste into `backend/.env`: `MEDGEMMA_RUNTIME=kaggle_tunnel`, `MEDGEMMA_TUNNEL_URL=...`. Restart the backend (`get_settings` is `@lru_cache(maxsize=1)` — `.env` changes need a restart).
4. `GET /api/v1/healthz` → tunnel reachable via the newly-wired `ping()`.
5. Submit one attempt per finding through the UI. Confirm feedback is genuinely image-specific (not the canned dry-run text), and that differentials and key observations now render distinctly.
6. `python -m pytest -m slow` — the live smoke test.

**Real eval:**
```powershell
python scripts\run_eval.py --runtime kaggle_tunnel --max-cases 30 --split test
```
Confirm `reports/phase1_eval.md` shows `Runtime: kaggle_tunnel`, non-zero p50/p95 latency, real token counts, and a structure validity figure. Keep `--max-cases` at 30 — Kaggle's 30 GPU-hr/week quota is the binding constraint (`DEVELOPMENT_PLAN.md:535`).

**Prerequisite to confirm before starting:** Hugging Face access to `google/medgemma-1.5-4b-it` is granted (it is gated), and an ngrok authtoken is available for `NGROK_AUTHTOKEN`.
