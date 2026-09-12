import type { ApiError } from "@/types/api";
import type { ToastInput, ToastVariant } from "@/components/ToastProvider";

const TOKEN_STORAGE_KEY = "cxt.accessToken";
const API_BASE = "/api/v1";

type TokenGetter = () => string | null;
type OnUnauthorized = () => void;
export type ToastEmitter = (toast: ToastInput) => void;

let getToken: TokenGetter = () => sessionStorage.getItem(TOKEN_STORAGE_KEY);
let onUnauthorized: OnUnauthorized = () => {
  sessionStorage.removeItem(TOKEN_STORAGE_KEY);
  window.location.assign("/login");
};
let emitToast: ToastEmitter | null = null;

export function configureApiClient(options: {
  tokenProvider?: TokenGetter;
  unauthorizedHandler?: OnUnauthorized;
  toastEmitter?: ToastEmitter | null;
}): void {
  if (options.tokenProvider !== undefined) {
    getToken = options.tokenProvider;
  }
  if (options.unauthorizedHandler !== undefined) {
    onUnauthorized = options.unauthorizedHandler;
  }
  if (options.toastEmitter !== undefined) {
    emitToast = options.toastEmitter;
  }
}

export class ApiClientError extends Error {
  public readonly httpStatus: number;
  public readonly code: string;

  constructor(httpStatus: number, code: string, message: string) {
    super(message);
    this.name = "ApiClientError";
    this.httpStatus = httpStatus;
    this.code = code;
  }
}

export function getStoredToken(): string | null {
  return sessionStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setStoredToken(token: string | null): void {
  if (token === null) {
    sessionStorage.removeItem(TOKEN_STORAGE_KEY);
  } else {
    sessionStorage.setItem(TOKEN_STORAGE_KEY, token);
  }
}

export interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
  body?: unknown;
  query?: QueryParams;
  signal?: AbortSignal;
  suppressErrorToast?: boolean;
}

export type QueryValue = string | number | boolean | null | undefined;
export type QueryParams = { [key: string]: QueryValue };

function variantForStatus(status: number): ToastVariant {
  if (status === 401) return "warning";
  if (status >= 500) return "error";
  if (status >= 400) return "warning";
  return "info";
}

function titleForError(status: number, code: string): string {
  if (status === 0) return "Cannot reach the server";
  if (status === 401) return "Please sign in again";
  if (status === 403) return "You do not have access";
  if (status === 404) return "Not found";
  if (status === 422) return "Invalid input";
  if (status === 409) return "Conflict";
  if (status >= 500) return "Server error";
  return code;
}

function detailForError(status: number, code: string, message: string): string {
  if (status === 0) {
    return "The backend is unreachable. Is the API server running on http://127.0.0.1:8000?";
  }
  if (code === "upstream_unavailable") {
    return message;
  }
  if (status >= 500) {
    return `${message} Please try again or check the server logs.`;
  }
  return message;
}

export async function apiRequest<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { method = "GET", body, query, signal, suppressErrorToast = false } = options;

  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  if (query !== undefined) {
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined && value !== null) {
        url.searchParams.set(key, String(value));
      }
    }
  }

  const headers: Record<string, string> = {
    Accept: "application/json",
  };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  const token = getToken();
  if (token !== null) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  let response: Response;
  try {
    response = await fetch(url.toString(), {
      method,
      headers,
      body: body === undefined ? null : JSON.stringify(body),
      ...(signal !== undefined ? { signal } : {}),
    });
  } catch (cause) {
    const err = new ApiClientError(
      0,
      "network_error",
      cause instanceof Error ? cause.message : "Network request failed.",
    );
    if (!suppressErrorToast) emitApiError(err);
    throw err;
  }

  if (response.status === 401) {
    if (!suppressErrorToast) emitToast?.({ variant: "warning", title: titleForError(401, "unauthorized") });
    onUnauthorized();
    throw new ApiClientError(401, "unauthorized", "Authentication required.");
  }

  if (!response.ok) {
    const payload = await safeReadError(response);
    const err = new ApiClientError(response.status, payload.code, payload.message);
    if (!suppressErrorToast) emitApiError(err);
    throw err;
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

function emitApiError(err: ApiClientError): void {
  if (emitToast === null) return;
  emitToast({
    variant: variantForStatus(err.httpStatus),
    title: titleForError(err.httpStatus, err.code),
    message: detailForError(err.httpStatus, err.code, err.message),
  });
}

async function safeReadError(response: Response): Promise<ApiError> {
  try {
    const data = (await response.json()) as Partial<ApiError>;
    return {
      code: typeof data.code === "string" ? data.code : "unknown_error",
      message:
        typeof data.message === "string"
          ? data.message
          : response.statusText,
      httpStatus: response.status,
    };
  } catch {
    return {
      code: "unknown_error",
      message: response.statusText || "Request failed.",
      httpStatus: response.status,
    };
  }
}
