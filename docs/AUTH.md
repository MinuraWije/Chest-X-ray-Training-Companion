# Auth — Roles, Token Flow, Threat Model

> Phase 1 of the CXT Companion. JWT (HS256) + argon2. Stateless tokens.
> No third-party SSO, no password reset, no email verification.

## Roles

| Role | Self-register? | Submit attempts | View own attempts | Review queue | Submit review |
|---|---|---|---|---|---|
| `STUDENT` | ✅ open | ✅ | ✅ | ❌ | ❌ |
| `EDUCATOR` | invite-only (by `ADMIN`) | ❌ | n/a | ✅ | ✅ |
| `ADMIN` | invite-only (by another `ADMIN`) | n/a | n/a | ✅ | ✅ |

The bootstrap `ADMIN` is created by `scripts/seed_admin.py` (Q6: invite-only for
non-`STUDENT` accounts). Subsequent `EDUCATOR` and `ADMIN` accounts are minted
via `POST /api/v1/auth/register` with `inviteAdminId` set to an existing
`ADMIN`'s id.

## Token flow

```
STUDENT                           Backend
   |                                 |
   |  POST /api/v1/auth/register     |
   |-------------------------------->|
   |                                 | argon2-cffi hashes password (time-constant verify)
   |                                 | SQLAlchemy INSERT users(id, email, password_hash, role)
   |                                 | PyJWT issue HS256({sub, role, iat, exp, iss, aud})
   |<----- 200 {accessToken, user}--|
   |                                 |
   |  POST /api/v1/attempts          |
   |  Authorization: Bearer <token>  |
   |-------------------------------->|
   |                                 | PyJWT decode(..., aud=cxt-companion, iss=cxt-companion)
   |                                 | users.get(sub)
   |                                 | use case executes
   |<------- 200 StudentAttempt------|
```

Tokens are stored in `sessionStorage` on the frontend (cleared on tab close;
re-login required). A 401 from any endpoint clears the token and redirects
to `/login`.

## Algorithm choices

- **Hashing:** `argon2id` (default `argon2-cffi` parameters; ~50 ms per
  verify on a modern CPU). OWASP's first-recommendation for password
  storage in 2024.
- **JWT:** HS256 with a 32+ byte secret. Symmetric because the only signer
  and verifier is this backend. Asymmetric (RS256) only buys us something
  when a separate service needs to verify without the signing key, which
  is not a Phase 1 concern.
- **Claims:** `sub` (user id), `role` (enum), `iat`, `exp`, `iss=cxt-companion`,
  `aud=cxt-companion`. The `iss` and `aud` are checked on every decode so
  a token issued for a sibling service cannot be replayed here.
- **TTL:** `AUTH_ACCESS_TOKEN_TTL_SECONDS=28800` (8 h) by default. Short
  enough to limit replay; long enough that a student does not get logged
  out during a study session.

## API

| Method | Path | Auth | Body | Returns |
|---|---|---|---|---|
| `POST` | `/api/v1/auth/register` | none | `{email, password, role?, inviteAdminId?}` | `TokenResponse` |
| `POST` | `/api/v1/auth/login` | none | `{email, password}` | `TokenResponse` |
| `GET`  | `/api/v1/auth/me` | bearer | – | `UserDto` |

`TokenResponse = { accessToken: string, user: UserDto }`.
`UserDto = { id, email, role, createdAt }`.

## Error contract

- `400` — not used (Pydantic does its own 422)
- `401` — missing or invalid bearer token
- `403` — valid token but the role is not allowed (`require_role(...)` failed)
- `409` — email already registered
- `422` — invalid input shape (Pydantic)

`DomainError` is mapped to the appropriate status by the exception handler
in `cxt.interfaces.api.app`. The frontend's `apiClient` reads
`{code, message}` and shows a toast.

## Threat model (Phase 1)

| Threat | Mitigation |
|---|---|
| Brute force on `/auth/login` | Phase 1: argon2 (~50 ms/verify) raises the cost. Phase 2+ should add a per-IP rate limit. |
| Stolen JWT | Short TTL (8 h). `sessionStorage` (not `localStorage`) limits persistence. Revoke on logout (client-side token discard; the token is still valid until TTL — acceptable for an educational prototype). |
| JWT secret leak | `.env`-only; never committed. `.env.example` ships a placeholder. Rotating `AUTH_JWT_SECRET` invalidates every outstanding token (one-time forced re-login). |
| Role escalation via `register` body | The use case explicitly checks `role`: only `STUDENT` is open; `EDUCATOR`/`ADMIN` require an existing `ADMIN`'s id. |
| CORS misconfig | Allow-list in `Settings.cors_allow_origins` (defaults to `http://127.0.0.1:5173` and `http://localhost:5173` only). No wildcard. |
| Token replay across services | `iss` + `aud` validated on every decode. |
| SQL injection | SQLAlchemy 2.0 parameterised queries. |
| Educational-bypass: a student can still submit attempts for cases they did not do | Out of scope for Phase 1 (the use case only needs the student's id; we are not tracking training-set membership yet). |

## Files

- `cxt/infrastructure/auth/jwt_auth_service.py` — argon2 + PyJWT.
- `cxt/infrastructure/data/sqlite_user_repository.py` — SQLAlchemy `IUserRepository`.
- `cxt/infrastructure/data/sqlite_attempt_repository.py` — SQLAlchemy `IAttemptRepository` (feedback stored as JSON).
- `cxt/infrastructure/data/sqlite_review_repository.py` — SQLAlchemy `IReviewRepository`.
- `cxt/application/use_cases/auth/__init__.py` — `RegisterUseCase`, `LoginUseCase`, `GetMeUseCase`.
- `cxt/application/use_cases/reviews.py` — `ListReviewQueueUseCase`, `ReviewAttemptUseCase`, `ListReviewsForAttemptUseCase`.
- `cxt/interfaces/api/routes/auth.py` — `/auth/register|login|me`.
- `cxt/interfaces/api/routes/reviews.py` — `/reviews/queue`, `/reviews/queue/{id}`, `/reviews/attempts/{id}`, `POST /reviews`.
- `cxt/interfaces/api/dependencies.py` — `current_user`, `require_role(...)`.
- `scripts/seed_admin.py` — bootstrap the first `ADMIN`.
- `backend/.env.example` — `AUTH_JWT_SECRET`, `AUTH_JWT_ALGORITHM`, `AUTH_JWT_ISSUER`, `AUTH_JWT_AUDIENCE`, `AUTH_ACCESS_TOKEN_TTL_SECONDS`, `ADMIN_EMAIL`, `ADMIN_PASSWORD`.
