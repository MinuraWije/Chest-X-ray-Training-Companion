# KAGGLE_RUNTIME — MedGemma inference server on Kaggle

The Kaggle notebook hosts a small FastAPI app that loads
`google/medgemma-1.5-4b-it` and exposes a `POST /generate`
endpoint over a public ngrok tunnel. The backend calls that tunnel
through `cxt.infrastructure.ml.kaggle_tunnel_client`.

## TL;DR

```python
# 1. Notebook — first cell
%pip install -q \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.32.0" \
    pydantic==2.9.2 \
    pyngrok==7.2.0 \
    transformers==4.46.0 \
    accelerate==1.0.0 \
    bitsandbytes==0.44.0 \
    pillow==10.4.0
!git clone https://github.com/<this repo>.git /kaggle/cxt

# 2. Notebook — second cell
import os
%cd /kaggle/cxt
from kaggle_runtime.medgemma_server import start_in_background

os.environ["MEDGEMMA_MODEL"] = "google/medgemma-1.5-4b-it"
os.environ["MEDGEMMA_PRECISION"] = "bf16"   # "bf16" | "fp32" | "8bit" | "4bit"
os.environ["NGROK_AUTHTOKEN"] = "<your token from https://dashboard.ngrok.com>"

# Loads the checkpoint FIRST, then opens the tunnel — the URL only appears
# once the model is actually ready to serve. Expect several minutes.
ngrok_url = start_in_background(port=8000)
print("MEDGEMMA_TUNNEL_URL =", ngrok_url)
# Keep the cell alive so the server stays up. In a real notebook, this
# is the last cell of the session; you can `time.sleep(...)` here.

# 3. Backend .env
MEDGEMMA_RUNTIME=kaggle_tunnel
MEDGEMMA_TUNNEL_URL=https://<the-url-above>.ngrok-free.app
MEDGEMMA_MODEL=google/medgemma-1.5-4b-it
```

The URL from step 2 is what the backend calls. **Restart the backend after
editing `.env`** — `get_settings()` is cached for the process lifetime.

> `%cd /kaggle/cxt`, not `/kaggle/cxt/kaggle_runtime`. The import is a
> package import and fails from inside the package directory.

> There is **no** `MEDGEMMA_DRY_RUN` setting. `MEDGEMMA_RUNTIME` is the only
> switch (`dry_run` | `kaggle_tunnel`); anything else is silently ignored.

## Files

| File | Purpose |
|---|---|
| `kaggle_runtime/medgemma_server.py` | FastAPI server + ngrok launcher. |
| `kaggle_runtime/prompts/system.txt` | System prompt (the JSON schema contract). |
| `kaggle_runtime/prompts/user.txt` | User prompt (carries the student's answer). |
| `backend/src/cxt/infrastructure/ml/kaggle_tunnel_client.py` | Backend HTTP client (`IMedGemmaClient` impl). |
| `backend/src/cxt/infrastructure/ml/image_loader.py` | Reads a case PNG and base64-encodes it. |
| `backend/src/cxt/infrastructure/ml/output_parser.py` | JSON extraction, repair, and `Feedback` validation. |
| `backend/src/cxt/infrastructure/ml/resilient_analyzer.py` | Corrective re-prompt on unparseable output. |
| `backend/src/cxt/infrastructure/ml/medgemma_analyzer.py` | `IImageAnalyzer` impl. |
| `backend/src/cxt/infrastructure/ml/dry_run_analyzer.py` | Deterministic canned feedback (offline / tests). |
| `backend/src/cxt/infrastructure/ml/prompts.py` | Backend prompt builder + prompt version. |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/ping` | Cheap health check. Returns `{"status":"ok"}`. |
| `GET` | `/healthz` | Pipeline-loaded status. Returns `{"status":"ok","pipeline_loaded":"True"}`. |
| `POST` | `/warmup` | Force the checkpoint to load. Idempotent; safe to call repeatedly. |
| `POST` | `/generate` | Run MedGemma on an image. Response: `{text, prompt_tokens, output_tokens, model}`. |

### `POST /generate` body

Provide **exactly one** image source; the request is rejected otherwise.

| Field | Notes |
|---|---|
| `image_base64` | Base64-encoded image bytes. **What the backend normally sends.** |
| `image_mime` | `image/png` (default) or `image/jpeg`. Used with `image_base64`. |
| `image_url_or_path` | A location the *Kaggle container* can open. Only useful for public URLs. |
| `prompt` | User-turn text (≤ 8000 chars). Carries the student's interpretation. |
| `system` | System-turn text (≤ 4000 chars). Carries the JSON schema instruction. |
| `max_new_tokens` | Default 300 server-side; the backend sends 512. |

**Why base64 and not a URL.** The manifest gives each case a browser-relative
`imageUrl` (`/images/00000013_023.png`) that resolves against the backend's
`StaticFiles` mount. Inside the Kaggle container that path does not exist, and
`127.0.0.1` there is the container itself — so a URL could only work if the
developer's own machine were *also* publicly tunnelled. Sending the bytes keeps
the system self-contained and puts nothing of the dev machine on the public
internet.

The student's free text goes in `prompt`, not `system`: it is user-turn content,
and the server's `system` cap is 4000 chars against `prompt`'s 8000.

## How the call works

```
Browser   Frontend  Backend  Kaggle  MedGemma
   |          |        |        |        |
   | submit   |        |        |        |
   |--------->|        |        |        |
   |          | POST   |        |        |
   |          | /attempts------>|        |
   |          |        | read   |        |
   |          |        | + b64  |        |
   |          |        | image  |        |
   |          |        | build  |        |
   |          |        | prompt |        |
   |          |        | POST   |        |
   |          |        | /generate------>|
   |          |        |        | temp   |
   |          |        |        | file   |
   |          |        |        | + run  |
   |          |        |        |<-------|
   |          |        | JSON   |        |
   |          |        |<-------|        |
   |          |        | parse  |        |
   |          |        | (repair/retry)  |
   |          |        | -> Feedback     |
   |          | JSON   |        |        |
   |          |<-------|        |        |
   | rendered |        |        |        |
   |<---------|        |        |        |
```

## Data flow and privacy

- The notebook receives the image bytes, the clinical context and the
  student's free-text interpretation. Bytes are written to a temporary file
  for the pipeline to read and deleted immediately after; they are never
  logged and never persisted. The server logs only the payload size, the MIME
  type and the prompt lengths.
- The images are de-identified public NIH ChestX-ray8 films, so no patient
  identifier ever leaves the machine.
- The student is identified by a UUID generated by the backend. The notebook
  never sees a student name or email.
- All traffic is over HTTPS because ngrok's free tier terminates TLS at the
  ngrok edge.
- For higher privacy in Phase 1.5 / Phase 2, swap the tunnel for a Cloud Run /
  Vertex AI endpoint with auth tokens; `KaggleTunnelMedGemmaClient` is one
  adapter and the use case does not care.

## Local dev (no Kaggle)

`MEDGEMMA_RUNTIME=dry_run` swaps in a `DryRunMedGemmaAnalyzer` that returns
canned feedback without touching the network. This is the default in
`backend/.env.example` and is what `pytest` and the dev server use. It also
swaps in a by-reference image loader, so dev runs do not read PNGs off disk
needlessly.

```bash
cd backend
python scripts/dev.py real-data
python scripts/dev.py run
# Open http://127.0.0.1:5173/cases/<id> and submit — feedback comes back
# in a few ms from the dry-run analyzer.
```

## Checking the runtime from the backend

`GET /api/v1/healthz` reports the configured runtime and, when it is
`kaggle_tunnel`, actually pings the tunnel:

```json
{"status": "degraded", "medgemmaRuntime": "kaggle_tunnel",
 "tunnelConfigured": true, "tunnelReachable": false,
 "hint": "MedGemma tunnel unreachable. Is the Kaggle notebook awake? ..."}
```

A configured URL is not a reachable one — restarting the notebook issues a new
ngrok URL, so the old value in `.env` goes stale silently.

## Precision

`MEDGEMMA_PRECISION` selects how the weights are loaded, and it is reported
back on `/healthz` so a run's numbers can be attributed to a precision:

| Value | Weights | Notes |
|---|---|---|
| `bf16` (default) | ~8.6GB | bfloat16, the model's training dtype. Fits a 16GB card. |
| `fp32` | ~17GB | Unambiguous fallback. Needs `GPU T4 x2`; `device_map="auto"` shards it. |
| `8bit` | ~4.6GB | If bf16 OOMs. |
| `4bit` | ~2.8GB | Smallest, most lossy. What Phase 1's first numbers were produced at. |
| `fp16` | ~8.6GB | **Broken — do not use.** Kept selectable only so the finding stays reproducible. |

**Never use `fp16` with this model.** Gemma-family models are trained in
bfloat16, whose exponent range matches float32's. float16's range is far
smaller, so the activations overflow to NaN, the model emits EOS immediately,
and every request returns HTTP 200 with an empty `text` — no exception, no
error log, nothing. A full 30-case evaluation completed at 0% before this was
noticed. bfloat16 is the correct half-precision choice even on Kaggle's
pre-Ampere cards, where torch runs it without hardware acceleration.

This is also why the `4bit` path was never affected: bitsandbytes defaults
`bnb_4bit_compute_dtype` to float32.

Quantisation is not free: it is a speed/memory trade against accuracy, so a
run's precision belongs next to its accuracy figure. The default was `4bit`
until the Phase 1 evaluation showed finding recognition barely above a six-way
chance baseline, at which point the quantisation error stopped being an
acceptable unknown.

`POST /warmup` generates once against a synthetic image and reports
`sanity_ok`. `run_eval.py` aborts when that check fails, so a misconfigured
dtype costs seconds rather than a 25-minute run and the GPU quota behind it.

`MEDGEMMA_USE_4BIT=0` is still honoured for older notebook cells and now means
`fp16`. It previously selected float32 with no `device_map`, which left the
model on the CPU — the flag was effectively unusable.

## Quota

Kaggle gives ~30 GPU hours / week. `Settings.max_eval_cases` (default 30) caps
the number of cases per eval run so demos and evaluations don't accidentally
burn the whole week's quota. `MEDGEMMA_RUNTIME=dry_run` runs the entire stack
offline.

## Verifying the tunnel by hand

```python
import base64, httpx

url = "https://<your>.ngrok-free.app"
headers = {"ngrok-skip-browser-warning": "true"}

print(httpx.get(f"{url}/ping", headers=headers).json())
print(httpx.post(f"{url}/warmup", headers=headers, timeout=600).json())

image_b64 = base64.b64encode(
    open("data/processed/images/00000013_023.png", "rb").read()
).decode()
r = httpx.post(
    f"{url}/generate",
    json={
        "image_base64": image_b64,
        "image_mime": "image/png",
        "prompt": "describe this chest x-ray in one sentence",
        "system": "you are a radiologist",
        "max_new_tokens": 200,
    },
    headers=headers,
    timeout=120,
)
print(r.status_code, r.json())
```

The `ngrok-skip-browser-warning` header matters: without it ngrok's free tier
may answer with an HTML interstitial, which surfaces as "tunnel returned
non-JSON". The backend client sets it on every request.

If you get `502 Bad Gateway` from ngrok, the notebook is asleep — wake it.

## Marked-slow smoke test

`tests/backend/unit/infrastructure/ml/test_kaggle_tunnel_client.py` stubs the
HTTP transport; it never calls Kaggle.

The real end-to-end check is
`tests/backend/integration/test_live_tunnel_smoke.py`. It is marked
`@pytest.mark.slow` and skips entirely unless `MEDGEMMA_TUNNEL_URL` is set, so
it cannot burn quota by accident:

```powershell
$env:MEDGEMMA_TUNNEL_URL = "https://<id>.ngrok-free.app"
python -m pytest -m slow -q
```

It pings the tunnel, warms the model, sends one real CXR from
`data/processed/images/`, and asserts the response parses to a valid `Feedback`
with a non-zero token count.

## Running the real evaluation

```powershell
python scripts\dev.py eval --runtime kaggle_tunnel --max-cases 30 --split test
```

`run_eval.py` pings the tunnel and calls `/warmup` before the first case, so
model-load time does not land in the latency percentiles (`--no-warmup` to skip).
Results go to `reports/phase1_eval.json` and `.md`, with `Runtime:
kaggle_tunnel` in the header. Keep `--max-cases` modest — the 30 GPU-hr/week
quota is the binding constraint.

## When the model's JSON is imperfect

A 4-bit 4B model does not always return clean JSON. Two layers handle that
before an attempt is failed:

1. **Static repair** in `OutputParser` — code fences (including unterminated
   ones), surrounding prose, trailing commas, typographic quotes, and output
   truncated at the token limit (the object is re-balanced and the dangling
   member dropped).
2. **Corrective re-prompt** in `RetryingFeedbackParser` — the unparseable
   output is quoted back to the model with the required keys restated
   (`MEDGEMMA_MAX_REPAIR_ATTEMPTS`, default 1).

Only if both fail does the attempt return 503. `feedback.raw` records
`prompt_version`, `model` and `repair_attempts` for every stored feedback, which
is what NFR-4 (traceability) requires.
