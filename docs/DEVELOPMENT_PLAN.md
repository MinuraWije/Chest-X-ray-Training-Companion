# Development Plan — Phase 1: MedGemma-Only Baseline

> **Project:** An AI-Assisted Chest X-ray Training Companion for Student Radiologists Using MedGemma and Retrieval-Augmented Generation
> **Phase:** 1 of 3 (MedGemma only → MedGemma + RAG → Fine-tuned MedGemma + RAG)
> **Purpose:** Reach the 50% development milestone — a working MedGemma inference pipeline + foundational UI — that satisfies the supervisor review requirement before thesis submission.

---

## 1. Decisions Locked In

| # | Decision | Choice | Implication |
|---|---|---|---|
| D1 | Dataset | **NIH Chest X-ray (ChestX-ray8)** is the Phase 1 primary. **MIMIC-CXR** is the Phase 1.5 fallback once PhysioNet credentialing is in place. | NIH is fully public — no credentialing, no licensing gates. MIMIC remains in the plan as the future primary because of richer free-text reports; the loader is one adapter so the swap is a one-file change. |
| D2 | UI framework | **React** (Vite + TypeScript) | Frontend decoupled from FastAPI; clean REST contract; affects Steps 6 and 7. |
| D3 | Findings | **All six** (pneumothorax, pleural effusion, cardiomegaly, consolidation / airspace opacity, nodule or mass, fracture) | Final working subset decided by label quality in Step 3. |
| D4 | Auth | **Role-based** (Student, Educator, Admin) | New auth domain + use cases + protected routes; affects Steps 2, 5, 6, 7. |
| D5 | MedGemma runtime | **Kaggle notebook + ngrok** as the **primary runtime**. No Google Health AI Developer Foundations API in Phase 1. | You already have a working MedGemma notebook on Kaggle. We add a thin FastAPI/Flask wrapper + ngrok inside the notebook so the project's FastAPI backend can call it as `IMedGemmaClient`. Free 30 hr/week T4/P100 GPU. |

---

## 2. Goals of Phase 1

| Goal | Measurable Outcome |
|---|---|
| G1 — End-to-end MedGemma-only pipeline | Student logs in → views a CXR → submits an attempt → receives structured feedback generated **without** retrieval. |
| G2 — Clean architecture foundation | Codebase follows SOLID + Clean Architecture. Adding RAG (Phase 2) requires **zero** changes to domain or use-case layers. |
| G3 — React frontend | A working SPA that talks to FastAPI, displays CXRs, captures attempts, and renders feedback sections. |
| G4 — Role-based auth | Three roles enforced (Student / Educator / Admin). Educator can review attempts; Admin manages the corpus and prompt versions. |
| G5 — Kaggle runtime | MedGemma inference runs in the Kaggle notebook, exposed via ngrok, called from the `MedGemmaAnalyzer` adapter. No local GPU required. |
| G6 — Foundational evaluation harness | A repeatable script that scores MedGemma-only outputs on finding recognition and structure validity. |
| G7 — Reproducible environment | `make install` sets up the backend (creates `.venv`, installs `requirements.txt`); `make install-fe` sets up the frontend; `.env.example` documents all keys. |

**Non-goals (explicit):** retrieval, fine-tuning, mobile UI, PACS integration, real patient data, password reset emails, third-party SSO, audit log UI (just DB table), cloud MedGemma billing setup.

---

## 3. Scope Boundaries

**In-scope (Phase 1)**
- Calling `google/medgemma-4b-it` running inside a **Kaggle notebook** (your existing one) via a thin `KaggleTunnelMedGemmaClient` adapter that POSTs to the ngrok-exposed URL.
- Loading a curated subset of **NIH Chest X-ray (ChestX-ray8)** with the six findings above (30–60 cases).
- Prompt engineering for the six findings, with the final working subset decided by label quality.
- Structured JSON output: likely finding, key signs, comparison with student answer, missed clues, learning point.
- React (Vite + TS) frontend with: login, case list, case view + attempt form, feedback view, educator review screen.
- FastAPI backend with JWT-based auth, RBAC middleware, and OpenAPI docs.
- SQLite-backed storage for users, cases, attempts, feedback, prompt versions.
- Pytest unit + integration tests; basic React tests.
- Foundational evaluation script.

**Out-of-scope (later phases)**
- Phase 1.5: MIMIC-CXR data swap once PhysioNet credentialing arrives (loader change only).
- Phase 2: vector DB, embeddings, RAG retriever, retrieval-relevance metric.
- Phase 3: LoRA / PEFT fine-tuning, fine-tuned evaluation, comparison write-up.
- Production deployment, CI/CD, container orchestration, password reset email, SSO.

---

## 4. Architecture — Clean Architecture + Hexagonal Ports

The codebase is organised so that **business rules do not depend on frameworks, UI, or cloud SDKs**. The Kaggle runtime, NIH data source, and the React UI are all adapters plugged into ports declared by the domain.

```
┌────────────────────────────────────────────────────────────────┐
│  frontend/  (React + Vite + TS, talks REST to backend)        │  ← delivery
└────────────────────────────┬───────────────────────────────────┘
                             │ HTTPS / JSON
┌────────────────────────────▼───────────────────────────────────┐
│  interfaces/api/  (FastAPI: routes, schemas, DI wiring)        │  ← delivery
│      │                                                           │
│      ▼                                                           │
│  application/  (Use cases incl. AuthService, AnalyzeCxr,        │  ← orchestration
│                 GenerateFeedback)                                │
│      │                                                           │
│      ▼                                                           │
│  domain/  (entities, value objects, PORTS — pure Python)         │  ← business rules
│      ▲                                                           │
│      │                                                           │
│  infrastructure/  (Kaggle MedGemma client, NIH loader,           │  ← adapters
│                    repos, config)                                 │
└────────────────────────────────────────────────────────────────┘
```

### 4.1 SOLID mapping

| Principle | How it shows up in this codebase |
|---|---|
| **S** — Single Responsibility | One class per concern: `MedGemmaAnalyzer` only orchestrates, `KaggleTunnelMedGemmaClient` only calls the tunnel, `NihLoader` only reads NIH, `JwtAuthService` only issues/verifies tokens. |
| **O** — Open/Closed | Phase 2 adds `VectorRetriever` next to `NullRetriever` (both implement `IRetriever`). Phase 1.5 swaps `NihLoader` for `MimicLoader` — no domain change. Phase 3 swaps `MedGemmaAnalyzer` for `FineTunedMedGemmaAnalyzer` — neither use case nor API changes. |
| **L** — Liskov Substitution | Any `IMedGemmaClient` (Kaggle, future Google API, future self-hosted) is interchangeable. Any `ICaseRepository` is interchangeable. |
| **I** — Interface Segregation | Narrow ports: `IMedGemmaClient.generate(image, prompt) -> ModelOutput`, `IUserRepository.find_by_email`, `IRetriever.retrieve`, `IFeedbackGenerator`. No fat interfaces. |
| **D** — Dependency Inversion | Use cases import `Protocol` ports, never `httpx`, `kaggle`, or `sqlalchemy`. Concrete wiring lives in `interfaces/api/app.py` (composition root). |

### 4.2 Clean code rules applied

- **Type hints everywhere**; `mypy --strict` on the backend, `tsc --strict` on the frontend.
- **No comments unless the *why* is non-obvious** (the "what" is the code's job).
- **Functions ≤ 30 lines**, **classes ≤ 200 lines**.
- **Naming carries intent**: `build_finding_prompt`, not `make_prompt_v2`.
- **No magic numbers / strings** — all knobs live in `infrastructure/config/settings.py` (Pydantic Settings).
- **Errors as values at boundaries, exceptions inside**: HTTP layer translates `DomainError` subclasses to status codes.
- **No global state** — settings, model client, DB session are injected.
- **Tests mirror source tree**.

---

## 5. Folder Layout

```
chest-xray-companion/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── Makefile                       # install, data, test, lint, eval, run
├── data/
│   ├── raw/                       # NIH Chest X-ray download (gitignored)
│   ├── processed/                 # curated subset, manifest CSV
│   └── corpus/                    # (Phase 2) RAG corpus
├── backend/
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── alembic/                   # (optional) DB migrations
│   └── src/cxt/
│       ├── domain/
│       │   ├── entities/
│       │   │   ├── case.py            # Case, CxrImage
│       │   │   ├── finding.py         # Finding, FindingCategory
│       │   │   ├── student_attempt.py
│       │   │   ├── feedback.py        # Feedback, FeedbackSection
│       │   │   ├── user.py            # User, Role
│       │   │   └── review.py          # EducatorReview
│       │   ├── value_objects/
│       │   │   ├── finding_label.py
│       │   │   ├── confidence.py
│       │   │   ├── rationale.py
│       │   │   └── email.py           # value-object Email
│       │   └── ports/
│       │       ├── image_analyzer.py      # IImageAnalyzer
│       │       ├── medgemma_client.py     # IMedGemmaClient
│       │       ├── retriever.py           # IRetriever (Null in Phase 1)
│       │       ├── feedback_generator.py  # IFeedbackGenerator
│       │       ├── case_repository.py
│       │       ├── attempt_repository.py
│       │       ├── user_repository.py
│       │       └── review_repository.py
│       ├── application/
│       │   ├── use_cases/
│       │   │   ├── auth/
│       │   │   │   ├── register.py
│       │   │   │   ├── login.py
│       │   │   │   └── require_role.py   # decorator / guard
│       │   │   ├── analyze_cxr.py
│       │   │   ├── generate_feedback.py
│       │   │   ├── list_cases.py
│       │   │   ├── review_attempt.py
│       │   │   └── list_reviews.py
│       │   ├── services/
│       │   │   └── jwt_service.py
│       │   └── dto/
│       │       ├── auth_dto.py
│       │       └── feedback_dto.py
│       ├── infrastructure/
│       │   ├── config/settings.py         # pydantic-settings
│       │   ├── ml/
│       │   │   ├── kaggle_tunnel_medgemma_client.py  # IMedGemmaClient impl
│       │   │   ├── medgemma_analyzer.py
│       │   │   ├── prompt_builder.py
│       │   │   └── output_parser.py
│       │   ├── retrieval/
│       │   │   └── null_retriever.py
│       │   ├── data/
│       │   │   ├── nih_loader.py
│       │   │   ├── manifest_repository.py     # CSV-backed CaseRepository
│       │   │   ├── sqlite_user_repository.py
│       │   │   ├── sqlite_attempt_repository.py
│       │   │   └── sqlite_review_repository.py
│       │   ├── auth/
│       │   │   ├── password_hasher.py         # argon2
│       │   │   └── jwt_auth_service.py
│       │   └── logging/setup.py
│       ├── interfaces/
│       │   ├── api/
│       │   │   ├── app.py                 # FastAPI factory + DI wiring
│       │   │   ├── deps.py                # current_user, require_role
│       │   │   ├── routes/
│       │   │   │   ├── auth.py
│       │   │   │   ├── cases.py
│       │   │   │   ├── attempts.py
│       │   │   │   ├── feedback.py
│       │   │   │   └── reviews.py
│       │   │   ├── middleware/
│       │   │   │   ├── cors.py
│       │   │   │   └── request_id.py
│       │   │   └── schemas/               # pydantic request/response
│       │   │       ├── auth.py
│       │   │       └── feedback.py
│       │   └── errors/
│       │       └── exception_handlers.py
│       └── shared/
│           ├── errors.py                  # DomainError hierarchy
│           └── types.py
├── kaggle_runtime/                    # separate folder for the notebook-side code
│   ├── medgemma_server.py              # FastAPI inside the notebook
│   ├── prompts/
│   │   └── system.txt
│   └── README.md                       # how to launch the notebook + tunnel
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── router.tsx                     # react-router
│       ├── api/
│       │   ├── client.ts                  # axios/fetch with JWT interceptor
│       │   ├── auth.ts
│       │   ├── cases.ts
│       │   ├── attempts.ts
│       │   └── reviews.ts
│       ├── auth/
│       │   ├── AuthContext.tsx
│       │   ├── ProtectedRoute.tsx
│       │   └── RoleGate.tsx
│       ├── pages/
│       │   ├── LoginPage.tsx
│       │   ├── RegisterPage.tsx
│       │   ├── CaseListPage.tsx
│       │   ├── CasePage.tsx               # CXR + attempt form
│       │   ├── FeedbackPage.tsx           # structured feedback
│       │   ├── ReviewQueuePage.tsx        # educator
│       │   └── ReviewPage.tsx             # educator rubric scoring
│       ├── components/
│       │   ├── CxrViewer.tsx
│       │   ├── AttemptForm.tsx
│       │   ├── FeedbackSection.tsx
│       │   ├── NavBar.tsx
│       │   └── EducationalBanner.tsx      # "for educational use only"
│       ├── hooks/
│       │   ├── useAuth.ts
│       │   └── useApi.ts
│       ├── types/
│       │   └── api.ts                     # mirrors backend DTOs
│       └── styles/
│           ├── globals.css
│           └── tokens.css
├── tests/
│   ├── conftest.py
│   ├── backend/
│   │   ├── unit/
│   │   │   ├── domain/
│   │   │   ├── application/
│   │   │   └── infrastructure/
│   │   ├── integration/
│   │   │   ├── test_api_auth.py
│   │   │   ├── test_api_cases.py
│   │   │   └── test_feedback_pipeline.py
│   │   └── golden/
│   │       ├── medgemma_response.json
│   │       └── feedback_dto.json
│   └── frontend/
│       ├── LoginPage.test.tsx
│       └── CasePage.test.tsx
├── notebooks/
│   ├── 01_nih_exploration.ipynb
│   ├── 02_prompt_experiments.ipynb
│   └── 03_evaluation.ipynb
├── scripts/
│   ├── download_nih_subset.py
│   ├── build_manifest.py
│   ├── seed_admin.py
│   └── run_eval.py
└── docs/
    ├── (existing) Research Idea, Proposal, Lit Review, etc.
    ├── DEVELOPMENT_PLAN.md                # this file
    ├── ARCHITECTURE.md                    # to be written in Step 1
    ├── API.md                             # auto-generated from OpenAPI
    ├── AUTH.md                            # roles, token flow
    ├── KAGGLE_RUNTIME.md                  # how to run the notebook server
    ├── DATA_NIH.md                        # source, license, schema, label noise
    ├── DATA_MIMIC.md                      # (Phase 1.5) credentialing, license, schema
    └── EVALUATION_RUBRIC.md
```

---

## 6. Tech Stack (locked in for Phase 1)

| Layer | Choice | Why |
|---|---|---|
| Backend language | Python 3.11+ | Matches proposal; modern typing. |
| Backend package manager | `pip` + `venv` (lockfile: `requirements.txt` + `requirements-dev.txt`) | Standard, no extra tooling. |
| Backend framework | FastAPI + Uvicorn | Proposal-aligned; typed routes, OpenAPI auto-gen, easy JWT middleware. |
| ORM / DB | SQLAlchemy 2.x + Alembic + SQLite | Standard; same code path can move to Postgres later. |
| Auth | JWT (PyJWT) + argon2 (pwd hash) | Stateless tokens, no session store; argon2 is the OWASP-recommended hasher. |
| Frontend | React 18 + Vite + TypeScript | Proposal-aligned; SPA gives better UX for the feedback sections. |
| Frontend routing | react-router-dom v6 | Standard. |
| Frontend data fetching | TanStack Query (React Query) | Cache + retry + loading states for free. |
| Frontend forms | react-hook-form + zod | Typed validation. |
| Frontend styling | Tailwind CSS | Fast for a single-developer prototype. |
| ML client (backend side) | `httpx`, `tenacity`, `pydantic` | Thin client that POSTs to the Kaggle tunnel; retries with backoff. |
| ML server (Kaggle side) | `transformers`, `bitsandbytes`, `accelerate`, `pyngrok`, FastAPI | Inside the Kaggle notebook only. |
| Model | `google/medgemma-1.5-4b-it` (4-bit) | The 1.5 release; instruction-tuned, multimodal, image-text-to-text. Fits T4. Invoked via `transformers.pipeline("image-text-to-text", ...)` (see Step 4). |
| Runtime | **Kaggle notebook + ngrok** | Free 30 hr/week T4/P100; existing working notebook. |
| Data | **NIH Chest X-ray (ChestX-ray8)** via `torchxrayvision` or direct NIH download | Public, de-identified, no licensing gate. |
| Data tooling | pandas, pyarrow, PIL | Standard. |
| Tests (backend) | `pytest`, `pytest-asyncio`, `httpx`, `pytest-cov` | Standard. |
| Tests (frontend) | Vitest + React Testing Library | Vite-native. |
| Lint/format | `ruff`, `black`, `mypy --strict` (BE); `eslint`, `prettier` (FE) | Single toolchain per side. |

---

## 7. Implementation Phases

Each step lists **deliverable**, **SOLID hook**, and **relative time**.

### Step 1 — Project scaffolding (½ day)

**Deliverable:** empty tree builds, `make test` runs, frontend `npm run dev` runs.

- `pyproject.toml` (project metadata only) + `requirements.txt` / `requirements-dev.txt` (pinned deps) + `package.json` (frontend).
- `ruff.toml`, `mypy.ini`, `.editorconfig`, `.gitignore`, `.env.example`.
- `Makefile` targets: `install`, `install-fe`, `data`, `seed`, `test`, `test-fe`, `lint`, `eval`, `run`, `run-fe`.
- `README.md` with quickstart for both halves.
- Empty `backend/src/cxt/` and `frontend/src/` packages.

**SOLID hook:** establishes directory boundaries that enforce the Dependency Rule.

### Step 2 — Domain model + ports (1 day)

**Deliverable:** pure-Python domain with no infra imports; ports declared as `Protocol`s.

Files: `domain/entities/*.py`, `domain/value_objects/*.py`, `domain/ports/*.py`, `shared/errors.py`.

Key shapes:
- `User(id, email, password_hash, role: Role, created_at)` with `Role` enum: `STUDENT | EDUCATOR | ADMIN`.
- `Case(id, image_url, ground_truth_finding, metadata)`.
- `StudentAttempt(case_id, student_id, free_text, submitted_at)`.
- `Feedback(sections, sources)` — `sources` is `[]` in Phase 1.
- `EducatorReview(attempt_id, educator_id, scores: dict, comment)`.
- Ports: `IImageAnalyzer`, `IMedGemmaClient`, `IRetriever` (Null in Phase 1), `ICaseRepository`, `IAttemptRepository`, `IUserRepository`, `IReviewRepository`, `IAuthService`.

**SOLID hook:** **D** — use cases import these Protocols, never `httpx` or `sqlalchemy`.

### Step 3 — NIH Chest X-ray data preparation (1 day)

**Deliverable:** `data/processed/manifest.csv` with 30–60 curated cases across the six findings, with corresponding image paths.

- `scripts/download_nih_subset.py` — downloads a small slice of NIH ChestX-ray (sample provided by NIH or via `torchxrayvision`). Records download provenance + checksums.
- `scripts/build_manifest.py` — joins images to NIH labels (the 14 "Cardiomegaly", "Effusion", "Pneumothorax", "Nodule", "Mass", "Fracture" labels map to our six findings), filters, writes `case_id,image_index,image_path,finding,label_source,split`.
- `ManifestRepository(CaseRepository)` — reads the CSV. `images/` are served by FastAPI via a `StaticFiles` mount on `/images`.
- Exploration notebook: class balance per finding, label noise notes (NIH labels are known to be noisy), sample image sanity check.

**NIH-specific notes:**
- NIH ChestX-ray8 / NIHCC labels come from NLP-mined radiology reports — they are noisy and not perfect ground truth. This is acknowledged in the manifest and the eval report.
- We use the 14 NIH disease labels and map them to our six target findings: `Pneumothorax → pneumothorax`, `Effusion → pleural effusion`, `Cardiomegaly → cardiomegaly`, `Consolidation + Infiltration + Pneumonia → consolidation / airspace opacity`, `Nodule + Mass → nodule or mass`, `Fracture → fracture`.
- A case with a positive label for a target finding is a positive case; a case with no positive label is a negative case.

**SOLID hook:** **O** — adding MIMIC later in Phase 1.5 is a new `MimicLoader` + new repo, no domain change.

### Step 4 — Kaggle runtime + MedGemma client (1 day)

**Deliverable:** the FastAPI backend can call MedGemma through a tunnel, end-to-end.

**Kaggle side (`kaggle_runtime/`):**
- `medgemma_server.py` — a small FastAPI app inside the notebook:
  - Loads `google/medgemma-1.5-4b-it` in 4-bit via `bitsandbytes`, then wraps it in `transformers.pipeline("image-text-to-text", model=..., image_processor=..., tokenizer=...)`.
  - `POST /generate` accepts `{image_url_or_path: str, prompt: str, system: str = ""}` and returns `{text: str, prompt_tokens: int, output_tokens: int}`. The server hands the URL/path straight to the pipeline (the pipeline supports both remote URLs and local file paths).
  - Uses `pyngrok` to expose port `8000` publicly; the URL is printed in the notebook output and pasted into the backend's `.env`.

  **Reference invocation (the same shape we use in the notebook):**
  ```python
  from transformers import pipeline

  pipe = pipeline("image-text-to-text", model="google/medgemma-1.5-4b-it")

  messages = [
      {
          "role": "user",
          "content": [
              {"type": "image", "url": "<CXR URL or local path>"},
              {"type": "text", "text": "<prompt>"},
          ],
      },
  ]
  result = pipe(text=messages)
  assistant_text = result[0]["generated_text"][-1]["content"]
  ```
  The `OutputParser` on the backend (and the JSON-extraction on the notebook side) reads `generated_text[-1]["content"]`, not the raw string.
- `prompts/system.txt` — default system prompt shared between notebook and backend.
- `README.md` — exact cells to run, what to copy where.

**Backend side (`backend/src/cxt/infrastructure/ml/`):**
- `KaggleTunnelMedGemmaClient(IMedGemmaClient)`:
  - Reads the ngrok URL from `Settings.medgemma_tunnel_url`.
  - POSTs image bytes (base64-encoded) + prompt with `tenacity` retry + exponential backoff (max 3 attempts).
  - Returns raw text + token usage.
  - Raises `MedGemmaUnavailable` on tunnel failure (lets the route handler return 503).
- `PromptBuilder` — composes the per-finding instruction from a template in `infrastructure/ml/prompts/`. Templates are plain text, easy to iterate in notebooks.
- `OutputParser` — extracts the assistant text from the pipeline's chat-format response (`result[0]["generated_text"][-1]["content"]`), strips markdown fences, parses the JSON, validates against the `Feedback` schema, raises `MalformedModelOutput` on failure (one retry with a corrective prompt).
- `MedGemmaAnalyzer(IImageAnalyzer)` — orchestrates build → call → parse.

**Kaggle-specific design points (see `docs/KAGGLE_RUNTIME.md`):**
- Tunnel URL is stored in `.env`, not in code.
- The notebook logs **only** image case-IDs, never image bytes.
- All requests go over HTTPS (ngrok provides this).
- A `Settings.medgemma_dry_run = true` flag swaps in a `FakeMedGemmaClient` for tests and offline demos.
- A short health-check endpoint on the notebook (`GET /ping`) lets the backend verify the tunnel is up before submitting a real request.

**SOLID hook:** **S** + **D** + **L** — UI and use cases know only `IImageAnalyzer`; the runtime (Kaggle now, Google API or self-hosted later) is one adapter.

### Step 5 — Use cases + auth (1–2 days)

**Deliverable:** seven use-case classes (auth: 3, CXR: 3, review: 1), each a single `execute(...)` method.

Auth use cases:
- `RegisterUseCase(email, password, role)` — validates, hashes with argon2, persists via `IUserRepository`. Default role is `STUDENT`; creating `EDUCATOR` or `ADMIN` requires a current `ADMIN` (enforced at the API layer).
- `LoginUseCase(email, password)` — verifies, returns a signed JWT (sub = user_id, role, exp).
- `RequireRoleUseCase` / decorator — guard utility, no side effects.

CXR use cases:
- `AnalyzeCxrUseCase(image, prompt) -> ModelOutput`.
- `GenerateFeedbackUseCase(case, attempt, student_id) -> Feedback`:
  1. Build prompt.
  2. `analyzer.analyze(image, prompt)`.
  3. Parse into `Feedback`.
  4. Persist via `IAttemptRepository` + `IFeedbackRepository` (one transaction).
- `ListCasesUseCase(split, current_user) -> list[CaseSummary]`.

Review use cases:
- `ReviewAttemptUseCase(attempt_id, scores, comment, educator_id) -> EducatorReview` — `EDUCATOR` or `ADMIN` only.
- `ListReviewsUseCase(filters) -> list[ReviewSummary]`.

**SOLID hook:** **L** — `IRetriever` is `NullRetriever`; `IMedGemmaClient` is the Kaggle impl; either is swappable without use-case changes.

### Step 6 — FastAPI + React UI (2–3 days)

**Deliverable:** running system at `http://localhost:5173` (frontend) talking to `http://localhost:8000` (backend).

**Backend (FastAPI):**
- `interfaces/api/app.py` — composition root (the only file that imports concretes).
- Routes (with `Depends(current_user)` + `Depends(require_role(...))`):
  - `POST /auth/register` — open.
  - `POST /auth/login` — open.
  - `GET  /cases` — any authenticated user.
  - `GET  /cases/{id}` — any authenticated user.
  - `POST /cases/{id}/attempts` — `STUDENT` only.
  - `GET  /attempts/{id}/feedback` — owner student + `EDUCATOR` + `ADMIN`.
  - `GET  /reviews/queue` — `EDUCATOR` + `ADMIN`.
  - `POST /attempts/{id}/reviews` — `EDUCATOR` + `ADMIN`.
- CORS middleware allowing the Vite dev origin.
- `StaticFiles` mount for `/images` (serves NIH JPEGs from `data/processed/images/`).
- Standard error handlers mapping `DomainError` → HTTP status.

**Frontend (React + Vite + TS):**
- `LoginPage` / `RegisterPage` — store JWT in `sessionStorage` (default for prototype).
- `ProtectedRoute` + `RoleGate` — guard pages by role.
- `CaseListPage` — list of cases with finding filter.
- `CasePage` — `CxrViewer` (scaled image), `AttemptForm` (textarea), submit button.
- `FeedbackPage` — render each `FeedbackSection` (`FeedbackSection` component) with collapsible evidence area (always empty in Phase 1, hidden if `[]`).
- `ReviewQueuePage` + `ReviewPage` (educator) — list attempts, apply rubric scores.
- Persistent `EducationalBanner` across all pages.
- `api/client.ts` — axios instance with `Authorization: Bearer <token>` interceptor and a 401 → redirect-to-login handler.

**SOLID hook:** **D** — composition root is the only place that knows concrete classes.

### Step 7 — Tests (1–2 days)

**Deliverable:** green `pytest` and `vitest`, ≥ 80% coverage on `domain/` and `application/`.

- Backend unit: domain entities (immutability, validation, role checks), `PromptBuilder` (golden outputs), `OutputParser` (incl. malformed), `NullRetriever`, `PasswordHasher`, `JwtAuthService`.
- Backend integration: full API with **fake** `IMedGemmaClient` and in-memory repos. Cover:
  - Register → login → token roundtrip.
  - Role guards (student cannot POST a review; unauth cannot call `/cases`).
  - Attempt submission → feedback retrieval.
- One backend smoke test that hits the real Kaggle tunnel on one NIH image (marked `@pytest.mark.slow`, skipped in CI by default).
- Frontend: render tests for `LoginPage` (validation), `CasePage` (form submit calls API with correct payload), `FeedbackPage` (renders all sections).

### Step 8 — Foundational evaluation harness (½ day)

**Deliverable:** `scripts/run_eval.py` produces `reports/phase1_eval.json` with:
- **Finding recognition**: % of cases where model's `likely_finding` matches the NIH label.
- **Structure validity**: % of outputs that parsed to a valid `Feedback`.
- **Per-finding breakdown** (since NIH labels are noisy, a flat accuracy is misleading).
- **Latency per case** (records p50 / p95).
- **Token usage** (records input + output tokens from the Kaggle client).

This is intentionally narrow in Phase 1; the full rubric (retrieval relevance, evidence grounding, learning usefulness) is filled in Phase 2.

### Step 9 — Documentation + handoff (½ day)

- `docs/ARCHITECTURE.md` — diagram + port/adaptor table.
- `docs/AUTH.md` — roles, token flow, threat model.
- `docs/KAGGLE_RUNTIME.md` — how to launch the notebook + ngrok, where to copy the URL, how to test the tunnel.
- `docs/DATA_NIH.md` — source, license, schema, label noise caveats, mapping table to our six findings.
- `docs/DATA_MIMIC.md` — (Phase 1.5 stub) credentialing process, license, schema.
- `docs/EVALUATION_RUBRIC.md` — first draft, copy of Appendix II from the proposal, updated with what Phase 1 actually measures.
- `README.md` — quickstart, how to run both halves, how to add a new finding, how to swap in a retriever (forward-looking note for Phase 2).
- `.env.example` complete with every variable.

---

## 8. 50% Milestone Checklist

The supervisor review requirement is met when **all** of the following are true:

- [ ] Repository builds from a clean clone (backend + frontend).
- [ ] Three roles exist and are enforced on the API: `STUDENT`, `EDUCATOR`, `ADMIN`.
- [ ] A student can register, log in, view at least one curated NIH case per finding, submit an attempt, and see structured feedback.
- [ ] `MedGemmaAnalyzer` runs end-to-end through the **Kaggle tunnel** client on at least one CXR per finding and produces a valid `Feedback`.
- [ ] At least 30 curated NIH cases are loaded and reachable from the UI (target subset size 30–60).
- [ ] Unit + integration tests are green; backend coverage ≥ 80% on `domain/` and `application/`.
- [ ] `scripts/run_eval.py` runs against the Kaggle runtime and reports Phase 1 metrics.
- [ ] Architecture is **RAG-ready** and **FT-ready**: swapping `NullRetriever` for a `VectorRetriever`, `NihLoader` for `MimicLoader`, or `MedGemmaAnalyzer` for `FineTunedMedGemmaAnalyzer` requires changing only the composition root and adding one adapter.
- [ ] `EducationalBanner` is visible on every page; auth tokens never logged; tunnel URL never logged.
- [ ] A short demo (screen recording + screenshots in `docs/screenshots/`) covers login, case attempt, feedback, and educator review.

---

## 9. What "Core Model Training" Means in Phase 1

Strictly, Phase 1 has **no training** — MedGemma runs in your existing Kaggle notebook, called over ngrok. The deliverable that maps onto the supervisor's "core model training" wording is the **core model pipeline**: Kaggle loader → tunnel → prompt → validate output on CXR images. This is the substrate that Phase 3 will later fine-tune.

If the supervisor specifically wants a training run before sign-off, the smallest possible substitute is a 50–100-step LoRA smoke test (which would still need a local GPU, since fine-tuning on the Kaggle notebook is possible but takes significant compute). It is **not** required for the 50% milestone but can be added in a half-day if asked.

---

## 10. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Kaggle tunnel URL changes / session dies** | High | On every backend startup, ping the tunnel `/ping` endpoint; if down, raise a clear `MedGemmaUnavailable` (HTTP 503) so the UI can show a "model warming up, retry" message. The user runs the notebook; the backend never silently fails. |
| **Kaggle 30 hr/week quota exhausted before demo** | Medium | Track usage; cap `Settings.max_eval_cases` (default 30); ship `medgemma_dry_run` so demos and dev work don't burn quota. If quota is the limiting factor, the evaluation can run on a 10–15 case subset. |
| **ngrok free tunnel URL changes on every restart** | High | Each notebook session prints the new URL; the user pastes it into `.env`. Documented in `KAGGLE_RUNTIME.md`. Acceptable for a single-user undergraduate prototype. |
| **Sending CXRs off-device via the tunnel** | Low–Medium | Data is de-identified NIH ChestX-ray (public, no patient identifiers); the privacy story is in `KAGGLE_RUNTIME.md`; educational-use banner makes the data flow transparent. |
| **NIH labels are noisy → accuracy metric is misleading** | High | Report per-finding breakdown; surface label source in the manifest; treat each label as "noisy NLP-mined ground truth" not "clinical truth". |
| **JSON-from-text output is unstable** | High | Robust parser + one retry with corrective prompt; strict schema; log raw outputs for offline review. |
| **JWT secret leak** | Medium | Read secret from env, never commit; rotate between dev/staging/prod; set short expiry; include `iss`, `aud`, `exp`, `iat`. |
| **Time slips on React polish** | Medium | Hard cap: one page per route, Tailwind utility classes, no custom design system. Cosmetic polish deferred. |
| **CORS / auth misconfig between Vite dev and FastAPI** | Low | Use `vite.config.ts` proxy in dev; explicit `CORSMiddleware` with allow-list in prod. |
| **MedGemma 4b-it gated on Hugging Face** | Low | Apply for access at the start of Step 4. If still pending, the `medgemma_dry_run` flag keeps the rest of the system testable. |

---

## 11. Phase 1.5, 2 & 3 Preview (for architectural context only — not built now)

- **Phase 1.5 — MIMIC-CXR swap:**
  - New `MimicLoader` and `MimicCaseRepository` (one adapter).
  - `ManifestRepository` swapped in composition root.
  - `docs/DATA_MIMIC.md` filled in with credentialing trail and license compliance notes.
  - No domain or use-case change.

- **Phase 2 — MedGemma + RAG:**
  - New `infrastructure/retrieval/vector_retriever.py` implementing `IRetriever`.
  - `data/corpus/` of curated teaching snippets (open-access).
  - `Feedback.sources` populated and rendered in `FeedbackSection` (collapsible "Show evidence" panel).
  - New rubric metrics: retrieval relevance, evidence grounding.
  - Single line change in composition root.

- **Phase 3 — Fine-tuned MedGemma + RAG:**
  - `infrastructure/ml/finetuned_analyzer.py` (PEFT/LoRA adapter on top of the same port).
  - `scripts/train_lora.py` (or notebook) — runs on a local GPU or a Kaggle GPU notebook.
  - New evaluation comparison report: MedGemma only vs +RAG vs +RAG+FT.
  - `ARCHITECTURE.md` updated with the training + serving loop.

---

## 12. Open Decisions — Resolved

| # | Question | Decision | Notes |
|---|---|---|---|
| Q1 | MedGemma runtime | **Kaggle notebook + ngrok** (primary). No Google Health AI Developer Foundations API in Phase 1. | You already have a working notebook. `IMedGemmaClient` is a port; `KaggleTunnelMedGemmaClient` is the only file that knows the runtime. A future `GoogleApiMedGemmaClient` can be added without touching use cases. |
| Q2 | MedGemma variant | **`google/medgemma-1.5-4b-it`** (lightweight, the 1.5 release) — confirmed. Invoked via `transformers.pipeline("image-text-to-text", ...)` with a chat-format messages list. | Can promote to `27b-multimodal` for the final eval if time / GPU budget allows. |
| Q3 | Dataset credentialing | **NIH Chest X-ray** primary (no credentialing). **MIMIC-CXR** as Phase 1.5 fallback once PhysioNet credentialing is in place. | Start the PhysioNet application in parallel with coding (free, ~1–2 weeks). The MIMIC swap is a one-file change in Step 3. |
| Q4 | Subset size | **30–60 cases** (target 30 to start; grow to 60 once Step 7 eval is green). | Matches the "as low as possible" Kaggle-quota constraint. |
| Q5 | Auth/roles in Phase 1 | **Required for Phase 1** (confirmed) | Student / Educator / Admin enforced at the API layer; minimal educator review screen in Phase 1. |
| Q6 | Self-registration policy | **Open** for `STUDENT`; invite-only for `EDUCATOR` / `ADMIN` | `scripts/seed_admin.py` creates the first admin. |
| Q7 | Educator review UI in Phase 1 | **Include a minimal review screen** (rubric from proposal Appendix II) | Saves re-doing auth work later. |
| Q8 | Image serving | **FastAPI `StaticFiles`** mount on `/images` | Simplest; document the upgrade to signed URLs in `docs/DATA_NIH.md`. |
| Q9 | Cost / quota | **As low as possible** — Kaggle free 30 hr/week. | Concrete guardrails: use `medgemma-4b-it`; cap `Settings.max_eval_cases` (default 30); ship `medgemma_dry_run` so dev work doesn't burn quota. Track token usage per request. |

### 12.1 Why Kaggle + ngrok and not local / not a paid cloud API

| Option | What it is | Pros | Cons | Phase 1 verdict |
|---|---|---|---|---|
| **Local (your machine)** | MedGemma runs on your own CPU/GPU | Full control, no time limits, persistent files | Needs your own GPU; image lives in your env | **Not used** in Phase 1 |
| **Kaggle notebook + ngrok** | Your existing notebook, exposed via ngrok tunnel | Free 30 hr/week T4/P100 GPU; you already have it working; reproducible; shareable; persistent Kaggle Datasets | Sessions die (≤12 h, 30 hr/week), ngrok URL changes on every restart | **Phase 1 primary runtime** |
| **Google Health AI Developer Foundations API** | Google's hosted MedGemma endpoint, REST | No GPU on our side, stable URL, no session limits | Pricing / free-tier not confirmed — cannot be assumed for an undergraduate project with a low budget | **Not used in Phase 1; revisit if supervisor requires it** |
| **Self-hosted cloud GPU VM** | Spin up a T4/L4 on a cloud provider, run MedGemma in a Docker container | Production-grade, stable URL | Costs money we are trying to avoid | **Not used** in Phase 1 |

**Conclusion:** Kaggle + ngrok is the right choice because (a) the notebook already works, (b) it is the only free option, (c) the `IMedGemmaClient` port lets us swap in a different runtime later without touching the use cases.

### 12.2 Why not Colab

| Dimension | Google Colab (free) | Kaggle (free) | Winner for this project |
|---|---|---|---|
| Free GPU time | Throttled after a few sessions; background-runners kick you off | 30 hr/week T4 x2 or P100, predictable | **Kaggle** |
| Session length | ≤ 12 h, often cut sooner under load | ≤ 12 h, generally more stable | Slight Kaggle |
| Persistent storage | Needs Google Drive setup | Built-in Kaggle Datasets (pin corpus + manifest) | **Kaggle** |
| Secret management | Colab secrets (basic) | `kaggle_secrets` (better for ngrok tokens) | Slight Kaggle |
| Existing work | None on this project | Your working MedGemma CXR notebook | **Kaggle** |
| Quota reset / transparency | Opaque cool-downs | Weekly reset, visible in the UI | **Kaggle** |
| Where Colab wins | Tighter Drive integration, broader community | — | Not relevant to this project |

**Conclusion:** if we ever need a notebook runtime besides Kaggle, Colab is not worth the context switch. Stick with Kaggle.

---

## 13. Next Steps

All open decisions are now resolved. Summary:

- **Dataset:** NIH Chest X-ray (ChestX-ray8), 30–60 cases across all six findings. MIMIC-CXR is the Phase 1.5 fallback once credentialing arrives.
- **UI:** React (Vite + TS) frontend; FastAPI backend; FastAPI `StaticFiles` for images.
- **Auth:** Student / Educator / Admin roles, JWT + argon2, open registration for `STUDENT`, invite-only for `EDUCATOR` / `ADMIN`, minimal educator review screen in Phase 1.
- **MedGemma:** `google/medgemma-4b-it` (lightweight). Runtime: **Kaggle notebook + ngrok** (primary; your existing notebook).
- **Quota / cost:** as low as possible — free Kaggle tier, `max_eval_cases=30`, `medgemma_dry_run` for dev.

**Concrete next actions for you (don't block on me):**
1. **Start the PhysioNet credentialing application now** (slowest blocker for Phase 1.5). Ask your supervisor whether he can share an already-downloaded MIMIC-CXR copy.
2. **Verify your Kaggle notebook still loads `google/medgemma-4b-it`** — apply for Hugging Face access if you don't have it yet.
3. **Send me the current prompt(s) and a sample output** from your Kaggle notebook so I can mirror them in the backend's `PromptBuilder`.

**What I'll do next (say the word and I'll start):**
1. **Step 1** — scaffold the backend (pip venv, `requirements.txt`, Makefile, `.env.example`, ruff/mypy, empty `cxt/` package) + scaffold the frontend (Vite + TS + Tailwind + react-router + TanStack Query).
2. **Step 2** — domain entities, value objects, and ports (`User`, `Role`, `Case`, `Finding`, `StudentAttempt`, `Feedback`, `IUserRepository`, `IMedGemmaClient`, `IRetriever`, `IFeedbackGenerator`, `IAuthService`, etc.) with unit tests.
3. Show you the diffs and check in before moving to Step 3.
