import { useQuery } from "@tanstack/react-query";
import { useParams, Link } from "react-router-dom";
import type { ReactNode } from "react";
import { getAttempt } from "@/api/cases";
import { AnswerComparison } from "@/components/AnswerComparison";
import { FeedbackSection, KEY_OBSERVATIONS_ID } from "@/components/FeedbackSection";
import { EducationalBanner } from "@/components/EducationalBanner";
import { caseTitle } from "@/utils/caseTitle";

export function FeedbackPage(): ReactNode {
  const { attemptId } = useParams<{ attemptId: string }>();
  const query = useQuery({
    queryKey: ["attempt", attemptId],
    queryFn: () => getAttempt(attemptId ?? ""),
    enabled: attemptId !== undefined,
  });

  if (query.isLoading) return <p className="p-6 text-slate-500">Loading…</p>;
  if (query.isError || query.data === undefined) {
    return (
      <p className="m-6 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
        {query.error instanceof Error ? query.error.message : "Failed to load."}
      </p>
    );
  }

  const attempt = query.data;
  const comparison = attempt.comparison ?? null;

  return (
    <div className="mx-auto max-w-6xl p-6">
      <EducationalBanner />
      <h1 className="mb-1 text-2xl font-semibold">Feedback for attempt</h1>
      <p className="mb-3 text-sm text-slate-600">
        <Link
          to={`/cases/${attempt.caseId}`}
          className="text-brand-600 hover:underline"
        >
          {caseTitle(attempt.caseId)}
        </Link>
      </p>
      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-2">
        {comparison !== null ? (
          <AnswerComparison
            freeText={attempt.freeText}
            comparison={comparison}
            whereToLookId={attempt.feedback !== null ? KEY_OBSERVATIONS_ID : undefined}
          />
        ) : (
          // An attempt whose case has left the manifest: show the answer
          // as written, without highlights.
          <div className="card p-3">
            <h2 className="text-sm font-semibold text-slate-900">Your interpretation</h2>
            <p className="mt-1 whitespace-pre-wrap text-sm text-slate-700">
              {attempt.freeText}
            </p>
          </div>
        )}
        {attempt.feedback !== null ? (
          <FeedbackSection feedback={attempt.feedback} />
        ) : (
          <p className="text-sm text-slate-500">No feedback available yet.</p>
        )}
      </div>
    </div>
  );
}
