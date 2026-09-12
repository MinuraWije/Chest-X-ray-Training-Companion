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
  source: string;
  text: string;
  score: number;
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
