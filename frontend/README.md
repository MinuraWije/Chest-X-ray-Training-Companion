# Frontend — Chest X-ray Training Companion

React 19 + Vite + TypeScript (strict) + Tailwind CSS + React Router v7 + TanStack Query v5 + react-hook-form + Zod.

## Current state (end of Step 5)

The SPA is fully wired: students can register/login, browse cases,
submit attempts, and read structured feedback. Educators and admins
can browse the review queue, open an attempt, and submit a rubric
review. The `AuthProvider` round-trips a JWT in `sessionStorage`; a
401 anywhere triggers a clear + redirect to `/login`.

| Frontend page | API call | Current behaviour |
|---|---|---|
| `GET /register` | `POST /api/v1/auth/register` | ✅ 200, stores token, redirects to `/cases` |
| `GET /login` | `POST /api/v1/auth/login` | ✅ 200, stores token, redirects to original page |
| `GET /cases` | `GET /api/v1/cases` | ✅ 200, lists all cases (300 with the real NIH subset, 35 with stub mode). Each card shows a thumbnail, `Case of {Label}` title, and a demographics sub-line. |
| `GET /cases/:id` | `GET /api/v1/cases/:id` | ✅ 200, shows the case + image viewer + same `Case of {Label}` + demographics header |
| `GET /cases/:id` submit | `POST /api/v1/attempts` | ✅ 200, shows structured `Feedback` |
| `GET /attempts/:id/feedback` | `GET /api/v1/attempts/:id` | ✅ 200 |
| `GET /reviews` | `GET /api/v1/reviews/queue` | ✅ 200 (educator/admin only; 403 otherwise) |
| `GET /reviews/:id` | `GET /api/v1/reviews/queue/:id` | ✅ 200, shows the attempt + feedback |
| `GET /reviews/:id` submit | `POST /api/v1/reviews` | ✅ 200, redirects back to `/reviews` |

## Prerequisites

- **Node.js 22+** and **npm 11+** (`node --version`, `npm --version`).
- The backend running on `http://127.0.0.1:8000` (see the top-level
  `README.md` and `backend/README.md`).

## Quick start (Windows / macOS / Linux)

```powershell
cd frontend
npm install            # one-time
npm run dev            # http://127.0.0.1:5173
```

That's it. Vite proxies `/api/*` to the backend on port 8000, so you do
not need to configure anything else.

**Use `127.0.0.1`, not `localhost`.** On Windows `localhost` resolves to
IPv6 first and Vite's default `localhost` binding can refuse the IPv4
request. The repo pins the dev server to `127.0.0.1` in `vite.config.ts`
to avoid this.

If you haven't started the backend yet, in another terminal:

```powershell
cd backend
python scripts\dev.py install-dev
python scripts\dev.py real-data                                 # 300-case real-data manifest
python scripts\dev.py run
```

> `real-data` is the one command to remember. It auto-detects
> `..\Dataset\images` + `..\Dataset\Data_Entry_2017_v2020.csv`,
> wipes the previous manifest, and rebuilds the 300-case
> real-data manifest. If you ever see the toast
> *"No curated case manifest found. From the backend folder run:
> python scripts\dev.py real-data"*, just run that command. See
> `backend/README.md` and the top-level `README.md` for the full
> real-data flow and override flags.

## Other scripts

```powershell
npm run build          # tsc -b + vite build (production bundle in dist/)
npm run preview        # serve the built bundle locally
npm run lint           # oxlint
npx tsc -b             # type-check only, no emit
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `EADDRINUSE :::5173` | Another Vite/dev server is already on that port. Close it or run `npm run dev -- --port 5174`. |
| `connect ECONNREFUSED ::1:8000` in the Vite logs | Vite was bound to IPv6 localhost. The repo pins Vite to `127.0.0.1` in `vite.config.ts` to avoid this. Don't override with `--host localhost`. |
| `502 Bad Gateway` from `/api/*` | Backend isn't running. Start it with `python scripts\dev.py run` from `backend/`. A red toast will say "Cannot reach the server". |
| `404 Not Found` from `/api/*` | Backend is running, but the route isn't implemented yet. See the table above. A red toast will say "Not found". |
| `404` on `/api/v1/cases` | Backend isn't running or `data` target hasn't been executed. Run `python scripts\dev.py real-data` (or `data --source stub --n-per-finding 5` for 35 synthetic cases), then restart the backend. |
| CORS errors in the browser console | The dev proxy in `vite.config.ts` already handles this. If you bypass the proxy, the CORS allowlist in `backend/src/cxt/interfaces/api/app.py` is `http://127.0.0.1:5173` and `http://localhost:5173`. |
| `npm install` fails on `vite` peer-dep warnings | Usually safe; check the actual error. We're on Vite 8 / React 19 / TS 6. |

## Error notifications

Every API failure shows a **toast** in the bottom-right corner so the
issue is obvious at a glance:

| Situation | Toast title | Detail |
|---|---|---|
| Backend down / proxy refused | **Cannot reach the server** | "The backend is unreachable. Is the API server running on http://127.0.0.1:8000?" |
| Vite proxy 502 Bad Gateway | **Cannot reach the server** | Same as above |
| 401 Unauthorized | **Please sign in again** | The token is cleared and the UI redirects to `/login` |
| 403 Forbidden | **You do not have access** | Server message |
| 404 Not found | **Not found** | Server message |
| 422 Validation error | **Invalid input** | Server message (often the form-field details) |
| 5xx | **Server error** | "Please try again or check the server logs." |
| `upstream_unavailable` (503) | **Server error** | Server message — usually *"No curated case manifest found. From the backend folder run: `python scripts\dev.py real-data`"* |

Toasts auto-dismiss after 5s (errors stay for 8s) and can be closed by
clicking the `×`. They are also accessible: errors are `role="alert"` /
`aria-live="assertive"`, non-errors are `role="status"` /
`aria-live="polite"`.

## Layout

```
src/
├── api/          fetch-based REST client + per-resource modules
├── auth/         AuthProvider, useAuth, ProtectedRoute, RoleGate
├── components/   NavBar, CxrViewer, AttemptForm, FeedbackSection,
│                 EducationalBanner, ToastProvider, ToastContainer
├── hooks/        (placeholder for future hooks)
├── pages/        one file per route — Login, Register, CaseList, Case,
│                 Feedback, ReviewQueue, Review
├── router.tsx    Route tree, role-gated
├── styles/       globals.css (Tailwind base + small design tokens)
├── types/        api.ts — TypeScript mirrors of backend DTOs
├── App.tsx       Providers (Query, Auth, Toast, Router)
└── main.tsx      createRoot
```

## Conventions

- All TypeScript in `strict` + `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes` + `noImplicitOverride`.
- `import type` for type-only imports (`verbatimModuleSyntax: true`).
- API base path: `/api/v1` (proxied to backend in dev).
- Token storage: `sessionStorage` under key `cxt.accessToken` (per D4 auth decision).
- Path alias: `@/*` → `src/*`.
- No comments unless the *why* is non-obvious (project rule).

## API expectations (current vs target)

**Currently implemented (Step 5):**

- `GET  /api/v1/` — route inventory
- `GET  /api/v1/healthz` — health check
- `GET  /api/v1/cases` — `Case[]` (filter: `split`, `finding`, `limit`, `offset`) — bearer required. Each `Case` has `id`, `imageUrl`, `clinicalContext`, `groundTruthCategory`, `groundTruthLabel`, `patientAge`, `patientSex`, `viewPosition`.
- `GET  /api/v1/cases/:id` — `Case` — bearer required (same fields)
- `POST /api/v1/auth/register` — `{ email, password, role?, inviteAdminId? }` → `{ accessToken, user }`
- `POST /api/v1/auth/login` — `{ email, password }` → `{ accessToken, user }`
- `GET  /api/v1/auth/me` — `UserDto` — bearer required
- `POST /api/v1/attempts` — `{ caseId, freeText }` → `StudentAttempt` (with `feedback`) — bearer, `STUDENT`
- `GET  /api/v1/attempts/:id` — `StudentAttempt` — bearer, owner-or-educator-or-admin
- `GET  /api/v1/attempts` — `StudentAttempt[]` — bearer; students see own, educators/admins see all
- `GET  /api/v1/reviews/queue` — `StudentAttempt[]` — bearer, `EDUCATOR`/`ADMIN`
- `GET  /api/v1/reviews/queue/:attemptId` — `StudentAttempt` (with `feedback`) — bearer, `EDUCATOR`/`ADMIN`
- `GET  /api/v1/reviews/attempts/:attemptId` — `EducatorReview[]` — bearer, `EDUCATOR`/`ADMIN`
- `POST /api/v1/reviews` — `{ attemptId, criteriaScores, comments }` → `EducatorReview` — bearer, `EDUCATOR`/`ADMIN`

All requests/responses are JSON. `401` triggers automatic logout + redirect to `/login`; `403` is surfaced as a toast.
