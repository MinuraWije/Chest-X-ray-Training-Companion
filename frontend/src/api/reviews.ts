import { apiRequest, type RequestOptions } from "@/api/client";
import type { EducatorReview, StudentAttempt } from "@/types/api";

export interface AttemptListResponse {
  items: StudentAttempt[];
  total: number;
}

export function listReviewQueue(
  options?: RequestOptions,
): Promise<AttemptListResponse> {
  return apiRequest<AttemptListResponse>("/reviews/queue", options);
}

export function getReview(
  attemptId: string,
  options?: RequestOptions,
): Promise<StudentAttempt> {
  return apiRequest<StudentAttempt>(`/reviews/queue/${attemptId}`, options);
}

export interface SubmitReviewRequest {
  attemptId: string;
  criteriaScores: Record<string, number>;
  comments: string;
}

export function submitReview(
  payload: SubmitReviewRequest,
  options?: RequestOptions,
): Promise<EducatorReview> {
  return apiRequest<EducatorReview>("/reviews", {
    method: "POST",
    body: payload,
    ...(options ?? {}),
  });
}
