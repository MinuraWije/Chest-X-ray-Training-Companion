# Backend — Chest X-ray Training Companion

FastAPI + Clean Architecture + strict mypy/ruff/pytest. Targets Python 3.11+.

## Current state

We are at the **end of Step 6 (foundational evaluation harness +
prompt-experiment notebook)**. The API boots, the manifest is
queryable, students can register/login, submit attempts, and get
structured `Feedback`. Educators and admins (invited by an existing
admin) can browse the review queue, fetch attempts, and submit
rubric-based reviews. SQLite (file-backed) stores users, attempts,
and reviews. `python scripts/run_eval.py` runs the Phase 1 metrics and
writes a JSON + Markdown report.

**The 50% development milestone is now complete.** This was the
supervisor review requirement before thesis submission.

### Live route inventory

The API exposes a self-describing route map at `GET /api/v1/`:

```bash
curl http://127.0.0.1:8000/api/v1/
```

Returns something like:

```json
{
  "name": "CXT Companion API",
  "version": "0.1.0",
  "docs": "/api/v1/docs",
  "openapi": "/api/v1/openapi.json",
  "phase": "Step 3 complete; Step 5 (auth/attempts/reviews) pending",
  "routes": [
    { "method": "GET", "path": "/api/v1/" },
    { "method": "GET", "path": "/api/v1/cases" },
    { "method": "GET", "path": "/api/v1/cases/{case_id}" },
    { "method": "GET", "path": "/api/v1/docs" },
    { "method": "GET", "path": "/api/v1/healthz" },
    { "method": "GET", "path": "/api/v1/openapi.json" }
  ]
}
```

If you try to hit an endpoint that isn't in that list, the API returns
`404 Not Found` — that's the expected behaviour right now, not a bug.

## Quick start (Windows / macOS / Linux)

The project ships with a cross-platform `scripts/dev.py` that does the
same things as the Makefile, so you do **not** need `make` installed.

```powershell
cd backend

# 1. One-time: create .venv, install runtime + dev deps, and `pip install -e .`
python scripts\dev.py install-dev

# 2. Build the curated case subset. `real-data` is the one command to
#    remember for the real-data workflow: it auto-detects
#    ..\Dataset\images + ..\Dataset\Data_Entry_2017_v2020.csv and
#    rebuilds the 300-case manifest from scratch.
python scripts\dev.py real-data                                 # 300 cases (5 findings x 50 + 50 negatives)
#    override the path / size if needed:
#    python scripts\dev.py real-data --local-dir C:\nih\images_001 --metadata-csv C:\nih\Data_Entry_2017.csv --n-per-finding 10

# 3. Start the API on http://127.0.0.1:8000 (auto-reload on file changes)
python scripts\dev.py run              # default host 127.0.0.1, port 8000
python scripts\dev.py run --host 0.0.0.0 --port 8000

# 4. Other targets
python scripts\dev.py test             # pytest (257 tests)
python scripts\dev.py lint             # ruff + mypy --strict
python scripts\dev.py format           # ruff --fix + black
python scripts\dev.py all              # install-dev + data --source stub + test + lint
python scripts\dev.py clean            # nuke caches and data/
```

### Real NIH data

`real-data` is the recommended entry point — it does the right thing
in one call:

```powershell
cd backend
python scripts\dev.py real-data
```

It wipes `data/raw` and `data/processed`, runs
`download_nih_subset --source local` against the bundled NIH subset,
then `build_manifest`. Defaults:
- `--local-dir` → `..\Dataset\images`
- `--metadata-csv` → `..\Dataset\Data_Entry_2017_v2020.csv`
- `--n-per-finding` → 50 (yields 300 cases; 5 findings × 50 + 50
  negatives; *fracture* is dropped because the NIH 14 NLP-mined
  labels don't include it).

If you unzipped the NIH box download elsewhere:

```powershell
python scripts\dev.py real-data `
    --local-dir C:\nih\images_001 `
    --metadata-csv C:\nih\Data_Entry_2017.csv `
    --n-per-finding 50
```

The manifest captures `patient_age`, `patient_sex`, `view_position`
from the metadata CSV and surfaces them via `/api/v1/cases` (as
`patientAge`, `patientSex`, `viewPosition`).

If the API ever returns `503` with `upstream_unavailable`, the
backend is up but the manifest is missing — just rerun
`real-data` (or, for a synthetic fallback while you wait for the
NIH download, `data --source stub --n-per-finding 5` for 35 stub
cases).

After `data` you can hit:
- `GET  /api/v1/` — route inventory.
- `GET  /api/v1/healthz` — health check.
- `GET  /api/v1/cases?split=train&finding=pneumothorax` — list cases
  (bearer). Returns `id`, `imageUrl`, `clinicalContext`,
  `groundTruthCategory`, `groundTruthLabel`, `patientAge`, `patientSex`,
  `viewPosition`.
- `GET  /api/v1/cases/{id}` — single case (same fields).
- `GET  /images/{filename}.png` — static image (FastAPI `StaticFiles`).
- `GET  /api/v1/docs` — Swagger UI (auto-generated from the OpenAPI schema).
- `GET  /api/v1/openapi.json` — OpenAPI schema.

## If you have `make` (Linux / macOS / WSL with make installed)

The `Makefile` exposes the same targets:

```bash
make install-dev
make data
make run
make test
make lint
make format
make all
make clean
```

## Manual fallback (no script, no make)

If even the script won't run (e.g. you want to wire it into an IDE):

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .

# data (real)
Remove-Item -Recurse -Force data\raw, data\processed -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m scripts.download_nih_subset --source local --local-dir ..\Dataset\images --metadata-csv ..\Dataset\Data_Entry_2017_v2020.csv --n-per-finding 50
.\.venv\Scripts\python.exe -m scripts.build_manifest --n-per-finding 50 --n-negatives 50
# (or the one-liner: python scripts\dev.py real-data)

# run
.\.venv\Scripts\python.exe -m uvicorn cxt.interfaces.api.app:app --host 127.0.0.1 --port 8000 --reload

# test / lint
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src scripts
```

On macOS / Linux the same commands work with `python3` and `.venv/bin/python`.

## Layout

```
backend/
├── src/cxt/                  the cxt-backend package (installed via `pip install -e .`)
│   ├── domain/               entities, value objects, ports — no framework imports
│   ├── application/          use cases, services, DTOs
│   ├── infrastructure/       adapters — config, ml, retrieval, data, auth, logging
│   ├── interfaces/           FastAPI surface
│   └── shared/               cross-cutting types and errors
├── scripts/                  operational scripts (download_nih_subset, build_manifest, dev, ...)
├── tests/                    pytest suite, mirrors src/cxt/ tree
├── data/                     created by `data` target; gitignored
│   ├── raw/                  PROVENANCE.json + raw images
│   └── processed/            manifest.csv + images/
├── docs/                     source docs (proposal, lit review, plan, etc.)
├── pyproject.toml            package metadata, ruff/mypy/pytest config
├── requirements.txt          runtime deps
├── requirements-dev.txt      + pytest, ruff, mypy, black
├── Makefile                  optional: same targets as scripts/dev.py
└── .env.example              full config surface (no secrets committed)
```

## Configuration

Copy `.env.example` to `.env` and adjust as needed. `MEDGEMMA_RUNTIME`
defaults to `dry_run` so dev work and demos do not burn Kaggle quota; set it
to `kaggle_tunnel` (with `MEDGEMMA_TUNNEL_URL`) for real MedGemma inference —
see `docs/KAGGLE_RUNTIME.md`. Settings are cached per process, so restart the
backend after editing `.env`.

## Conventions

- `mypy --strict` everywhere.
- `ruff check` linting (E/F/I/N/W/UP/B/SIM/RUF).
- Functions ≤ 30 lines, classes ≤ 200 lines, no global state.
- No comments unless the *why* is non-obvious.
- All paths absolute; project root is computed from `__file__` so the
  service can be run from anywhere.

## What's next

- **Phase 1.5** — MIMIC-CXR data swap (one adapter + loader change;
  no domain change). The eval harness re-runs against the MIMIC
  subset; the per-finding breakdown is the table to read.
- **Phase 2** — add a `VectorRetriever` (a one-line change in the
  composition root). Extend the eval harness with retrieval-relevance
  and evidence-grounding metrics. See `docs/EVALUATION_RUBRIC.md`.
- **Phase 3** — fine-tune with LoRA / PEFT. The eval harness stays
  the same; the analyzer gets swapped for a `FineTunedMedGemmaAnalyzer`.
