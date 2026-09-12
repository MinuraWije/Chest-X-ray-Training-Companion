import { apiRequest, type QueryParams, type RequestOptions } from "@/api/client";
import type { Case, FindingCategory, StudentAttempt } from "@/types/api";

export interface ListCasesQuery {
  category?: FindingCategory;
  page?: number;
  pageSize?: number;
  [key: string]: QueryParams[string];
}

export interface CaseListResponse {
  items: Case[];
  total: number;
}

export function listCases(
  query: ListCasesQuery = {},
  options?: RequestOptions,
): Promise<CaseListResponse> {
  return apiRequest<CaseListResponse>("/cases", { query, ...(options ?? {}) });
}

export function getCase(caseId: string, options?: RequestOptions): Promise<Case> {
  return apiRequest<Case>(`/cases/${caseId}`, options);
}

export interface SubmitAttemptRequest {
  caseId: string;
  freeText: string;
}

export function submitAttempt(
  payload: SubmitAttemptRequest,
  options?: RequestOptions,
): Promise<StudentAttempt> {
  return apiRequest<StudentAttempt>("/attempts", {
    method: "POST",
    body: payload,
    ...(options ?? {}),
  });
}

export function getAttempt(
  attemptId: string,
  options?: RequestOptions,
): Promise<StudentAttempt> {
  return apiRequest<StudentAttempt>(`/attempts/${attemptId}`, options);
}
