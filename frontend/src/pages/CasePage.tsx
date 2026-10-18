import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { useState, type ReactNode } from "react";
import { getCase, submitAttempt } from "@/api/cases";
import type { StudentAttempt } from "@/types/api";
import { CxrViewer } from "@/components/CxrViewer";
import { AttemptForm, type AttemptFormValues } from "@/components/AttemptForm";
import { AnswerComparison } from "@/components/AnswerComparison";
import { FeedbackSection, KEY_OBSERVATIONS_ID } from "@/components/FeedbackSection";
import { EducationalBanner } from "@/components/EducationalBanner";
import { caseTitle } from "@/utils/caseTitle";

export function CasePage(): ReactNode {
  const { caseId } = useParams<{ caseId: string }>();
  const queryClient = useQueryClient();
  const [attempt, setAttempt] = useState<StudentAttempt | null>(null);

  const caseQuery = useQuery({
    queryKey: ["case", caseId],
    queryFn: () => getCase(caseId ?? ""),
    enabled: caseId !== undefined,
  });

  const submitMutation = useMutation({
    mutationFn: (values: AttemptFormValues) => {
      if (caseId === undefined) throw new Error("Missing case id.");
      return submitAttempt({ caseId, freeText: values.freeText });
    },
    onSuccess: (submitted) => {
      setAttempt(submitted);
      void queryClient.invalidateQueries({ queryKey: ["case", caseId] });
    },
  });

  if (caseQuery.isLoading) {
    return <p className="p-6 text-slate-500">Loading case…</p>;
  }
  if (caseQuery.isError || caseQuery.data === undefined) {
    return (
      <p className="m-6 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
        {caseQuery.error instanceof Error
          ? caseQuery.error.message
          : "Failed to load case."}
      </p>
    );
  }

  const c = caseQuery.data;

  const demoParts: string[] = [];
  if (c.patientAge) demoParts.push(`${c.patientAge}-year-old`);
  if (c.patientSex) {
    const sex = c.patientSex.toUpperCase();
    if (sex === "M") demoParts.push("male");
    else if (sex === "F") demoParts.push("female");
    else demoParts.push(sex.toLowerCase());
  }
  if (c.viewPosition) demoParts.push(`${c.viewPosition.toUpperCase()} view`);
  const demoLine = demoParts.length > 0 ? demoParts.join(" \u00b7 ") : null;

  return (
    <div className="mx-auto max-w-6xl p-6">
      <EducationalBanner />
      <h1 className="mb-1 text-2xl font-semibold">
        {caseTitle(c.id)}
      </h1>
      {demoLine !== null && (
        <p className="mb-4 text-sm text-slate-600">{demoLine}</p>
      )}
      {c.clinicalContext && (
        <p className="mb-4 text-sm text-slate-600">{c.clinicalContext}</p>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <CxrViewer
          src={c.imageUrl}
          alt={`Chest radiograph, ${caseTitle(c.id)}`}
        />
        <div className="space-y-4">
          {attempt === null ? (
            <AttemptForm
              submitting={submitMutation.isPending}
              onSubmit={async (values) => {
                await submitMutation.mutateAsync(values);
              }}
            />
          ) : (
            <>
              {attempt.comparison != null && (
                <AnswerComparison
                  freeText={attempt.freeText}
                  comparison={attempt.comparison}
                  whereToLookId={attempt.feedback !== null ? KEY_OBSERVATIONS_ID : undefined}
                />
              )}
              {attempt.feedback !== null ? (
                <FeedbackSection feedback={attempt.feedback} />
              ) : (
                <p className="text-sm text-slate-500">No feedback available yet.</p>
              )}
            </>
          )}
          {submitMutation.isError && (
            <p className="rounded-md border border-red-200 bg-red-50 p-2 text-sm text-red-700">
              {submitMutation.error instanceof Error
                ? submitMutation.error.message
                : "Submission failed."}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
