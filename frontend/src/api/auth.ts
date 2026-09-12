import { apiRequest, setStoredToken, type RequestOptions } from "@/api/client";
import type { AuthUser, Role, TokenPair } from "@/types/api";

interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  role?: Role;
  inviteAdminId?: string;
}

interface MeResponse {
  id: string;
  email: string;
  role: Role;
  createdAt: string;
}

function toAuthUser(payload: MeResponse): AuthUser {
  return { id: payload.id, email: payload.email, role: payload.role };
}

function pickOptions(extras: RequestOptions | undefined): RequestOptions {
  return extras ?? {};
}

export async function login(
  payload: LoginRequest,
  options?: RequestOptions,
): Promise<AuthUser> {
  const pair = await apiRequest<TokenPair>("/auth/login", {
    method: "POST",
    body: payload,
    ...pickOptions(options),
  });
  setStoredToken(pair.accessToken);
  return me(
    options?.suppressErrorToast !== undefined
      ? { suppressErrorToast: options.suppressErrorToast }
      : undefined,
  );
}

export async function register(
  payload: RegisterRequest,
  options?: RequestOptions,
): Promise<AuthUser> {
  const pair = await apiRequest<TokenPair>("/auth/register", {
    method: "POST",
    body: { role: "STUDENT", ...payload },
    ...pickOptions(options),
  });
  setStoredToken(pair.accessToken);
  return me(
    options?.suppressErrorToast !== undefined
      ? { suppressErrorToast: options.suppressErrorToast }
      : undefined,
  );
}

export async function me(options?: RequestOptions): Promise<AuthUser> {
  const mePayload = await apiRequest<MeResponse>("/auth/me", {
    ...(options ?? {}),
  });
  return toAuthUser(mePayload);
}

export function logout(): void {
  setStoredToken(null);
}
