import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import type { ReactNode } from "react";
import { listReviewQueue } from "@/api/reviews";
import { EducationalBanner } from "@/components/EducationalBanner";

export function ReviewQueuePage(): ReactNode {
  const query = useQuery({
    queryKey: ["reviews", "queue"],
    queryFn: listReviewQueue,
  });

  const items = query.data?.items ?? [];
  const total = query.data?.total ?? 0;

  return (
    <div className="mx-auto max-w-5xl p-6">
      <EducationalBanner />
      <h1 className="mb-1 text-2xl font-semibold">Review queue</h1>
      <p className="mb-4 text-sm text-slate-600">
        Student attempts waiting for educator review.
      </p>

      {query.isLoading && <p className="text-slate-500">Loading…</p>}
      {query.isError && (
        <p className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {query.error instanceof Error ? query.error.message : "Failed to load."}
        </p>
      )}

      {!query.isLoading && !query.isError && total > 0 && (
        <p className="mb-3 text-xs text-slate-500">
          {total} attempt{total === 1 ? "" : "s"} waiting.
        </p>
      )}

      {!query.isLoading && !query.isError && total === 0 && (
        <p className="text-slate-500">No attempts waiting for review.</p>
      )}

      {items.length > 0 && (
        <ul className="space-y-2">
          {items.map((a) => (
            <li key={a.id} className="card p-3">
              <Link to={`/reviews/${a.id}`} className="block hover:underline">
                <div className="text-sm font-semibold text-slate-900">
                  Attempt {a.id}
                </div>
                <div className="text-xs text-slate-500">
                  Case {a.caseId} · {new Date(a.submittedAt).toLocaleString()}
                </div>
                <p className="mt-1 line-clamp-2 text-sm text-slate-700">
                  {a.freeText}
                </p>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
