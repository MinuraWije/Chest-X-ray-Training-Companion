export type Role = "STUDENT" | "EDUCATOR" | "ADMIN";

export type FindingCategory =
  | "pneumothorax"
  | "pleural_effusion"
  | "cardiomegaly"
  | "consolidation"
  | "nodule_or_mass"
  | "fracture"
  | "no_finding";

export interface AuthUser {
  id: string;
  email: string;
  role: Role;
}

export interface TokenPair {
  accessToken: string;
}

export interface Finding {
  category: FindingCategory;
  label: string;
  confidence: number;
  rationale: string;
  source: "model" | "ground_truth";
}

export interface RetrievedPassage {
  /**
   * Stable corpus chunk id. Key on this, never on `source`: one document
   * contributes many chunks and they all carry the same `source` string.
   */
  id: string;
  source: string;
  url: string;
  text: string;
  /**
   * Absolute, saturating retrieval score in [0, 1). Deliberately not shown to
   * students as a percentage — see `SourceList` in FeedbackSection.tsx.
   */
  score: number;
  /**
   * Curated student-facing "Learn more" link (verified teaching article).
   * Empty when no verified link exists for the chunk — the UI hides the
   * link instead of pointing anywhere unverified.
   */
  readMoreUrl: string;
  readMoreLabel: string;
}

export interface FeedbackSection {
  heading: string;
  body: string;
}

export interface Feedback {
  primaryFinding: Finding;
  differentialFindings: Finding[];
  keyObservations: string;
  teachingPoint: string;
  sources: RetrievedPassage[];
  raw: Record<string, unknown>;
}

export interface Case {
  id: string;
  imageUrl: string;
  clinicalContext: string;
  groundTruthCategory: FindingCategory;
  groundTruthLabel: string;
  patientAge: string;
  patientSex: string;
  viewPosition: string;
}

export interface StudentAttempt {
  id: string;
  caseId: string;
  studentId: string;
  freeText: string;
  submittedAt: string;
  feedback: Feedback | null;
}

export interface EducatorReview {
  id: string;
  attemptId: string;
  educatorId: string;
  criteriaScores: Record<string, number>;
  comments: string;
  createdAt: string;
}

export interface ApiError {
  code: string;
  message: string;
  httpStatus: number;
}
