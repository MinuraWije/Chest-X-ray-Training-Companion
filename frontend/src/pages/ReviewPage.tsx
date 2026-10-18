import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useParams, useNavigate } from "react-router-dom";
import { useState, type ReactNode } from "react";
import { getReview, submitReview } from "@/api/reviews";
import { EducationalBanner } from "@/components/EducationalBanner";

// These six keys are the rubric the backend accepts: `_VALID_CRITERIA` in
// `backend/src/cxt/domain/entities/review.py`. The domain rejects any other
// key, so a form that sends its own names produces HTTP 422 on every
// submission — which is what this page did until the two lists were aligned.
// `test_review_form_rubric.py` fails if they drift apart again.
//
// The educator scores the *model's feedback*, not the student's answer: that
// is what FR-7 asks an educator to validate.
const RUBRIC = [
  "finding_recognition",
  "retrieval_relevance",
  "evidence_grounding",
  "clarity",
  "usefulness_for_learning",
  "safety_trustworthiness",
] as const;

type Criterion = (typeof RUBRIC)[number];

const CRITERION_LABELS: Record<Criterion, string> = {
  finding_recognition: "Finding recognition",
  retrieval_relevance: "Relevance of retrieved material",
  evidence_grounding: "Grounding in the supplied evidence",
  clarity: "Clarity",
  usefulness_for_learning: "Usefulness for learning",
  safety_trustworthiness: "Safety and trustworthiness",
};

type Scores = Record<Criterion, number>;

function emptyScores(): Scores {
  return {
    finding_recognition: 3,
    retrieval_relevance: 3,
    evidence_grounding: 3,
    clarity: 3,
    usefulness_for_learning: 3,
    safety_trustworthiness: 3,
  };
}

export function ReviewPage(): ReactNode {
  const { attemptId } = useParams<{ attemptId: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [scores, setScores] = useState<Scores>(() => emptyScores());
  const [comments, setComments] = useState<string>("");

  const attemptQuery = useQuery({
    queryKey: ["review", attemptId],
    queryFn: () => getReview(attemptId ?? ""),
    enabled: attemptId !== undefined,
  });

  const submitMutation = useMutation({
    mutationFn: () => {
      if (attemptId === undefined) throw new Error("Missing attempt id.");
      return submitReview({
        attemptId,
        criteriaScores: scores,
        comments,
      });
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["reviews", "queue"] });
      navigate("/reviews", { replace: true });
    },
  });

  if (attemptQuery.isLoading) return <p className="p-6 text-slate-500">Loading…</p>;
  if (attemptQuery.isError || attemptQuery.data === undefined) {
    return (
      <p className="m-6 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
        {attemptQuery.error instanceof Error
          ? attemptQuery.error.message
          : "Failed to load."}
      </p>
    );
  }

  const a = attemptQuery.data;

  return (
    <div className="mx-auto max-w-3xl p-6">
      <EducationalBanner />
      <h1 className="mb-1 text-2xl font-semibold">Review attempt {a.id}</h1>
      <p className="mb-3 text-sm text-slate-600">Case {a.caseId}</p>

      <div className="card mb-4 p-3">
        <h2 className="text-sm font-semibold text-slate-900">Student interpretation</h2>
        <p className="mt-1 whitespace-pre-wrap text-sm text-slate-700">
          {a.freeText}
        </p>
      </div>

      {a.feedback !== null && (
        <div className="card mb-4 p-3">
          <h2 className="text-sm font-semibold text-slate-900">Model feedback given</h2>
          <p className="mt-1 text-sm text-slate-700">
            Primary: {a.feedback.primaryFinding.label} ({(a.feedback.primaryFinding.confidence * 100).toFixed(0)}%)
          </p>
          <p className="mt-1 text-sm text-slate-700">{a.feedback.teachingPoint}</p>
        </div>
      )}

      <form
        className="card p-4"
        onSubmit={(e) => {
          e.preventDefault();
          void submitMutation.mutate();
        }}
      >
        <h2 className="mb-3 text-lg font-semibold text-slate-900">
          Your review of the feedback above
        </h2>
        <div className="space-y-3">
          {RUBRIC.map((criterion) => (
            <div key={criterion} className="flex items-center justify-between">
              <label className="text-sm text-slate-700" htmlFor={criterion}>
                {CRITERION_LABELS[criterion]}
              </label>
              <input
                id={criterion}
                type="range"
                min={1}
                max={5}
                step={1}
                value={scores[criterion]}
                onChange={(e) =>
                  setScores((s) => ({ ...s, [criterion]: Number(e.target.value) }))
                }
                className="w-1/2"
              />
              <span className="w-6 text-right tabular-nums text-sm text-slate-700">
                {scores[criterion]}
              </span>
            </div>
          ))}
        </div>

        <label className="label mt-4" htmlFor="comments">Comments</label>
        <textarea
          id="comments"
          className="input min-h-[120px]"
          value={comments}
          onChange={(e) => setComments(e.target.value)}
        />

        {submitMutation.isError && (
          <p className="mt-3 rounded-md border border-red-200 bg-red-50 p-2 text-sm text-red-700">
            {submitMutation.error instanceof Error
              ? submitMutation.error.message
              : "Submission failed."}
          </p>
        )}

        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            className="btn-ghost"
            onClick={() => navigate("/reviews")}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="btn-primary"
            disabled={submitMutation.isPending}
          >
            {submitMutation.isPending ? "Submitting…" : "Submit review"}
          </button>
        </div>
      </form>
    </div>
  );
}
