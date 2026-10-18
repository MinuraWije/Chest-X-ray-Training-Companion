import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import type { ReactNode } from "react";
import { listCases } from "@/api/cases";
import { EducationalBanner } from "@/components/EducationalBanner";
import type { Case } from "@/types/api";
import { caseTitle } from "@/utils/caseTitle";

function demographicsLine(c: Case): string | null {
  const parts: string[] = [];
  if (c.patientAge) {
    const age = Number(c.patientAge);
    parts.push(
      Number.isFinite(age) ? `${age}-year-old patient` : `${c.patientAge}-year-old patient`,
    );
  }
  if (c.patientSex) {
    const sex = c.patientSex.toUpperCase();
    if (sex === "M") parts.push("male");
    else if (sex === "F") parts.push("female");
    else parts.push(sex.toLowerCase());
  }
  if (c.viewPosition) {
    parts.push(`${c.viewPosition.toUpperCase()} view`);
  }
  return parts.length > 0 ? parts.join(" \u00b7 ") : null;
}

export function CaseListPage(): ReactNode {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["cases"],
    queryFn: () => listCases(),
  });

  const cases = data?.items ?? [];
  const total = data?.total ?? 0;

  return (
    <div className="mx-auto max-w-6xl p-6">
      <EducationalBanner />
      <h1 className="mb-1 text-2xl font-semibold">Available cases</h1>
      <p className="mb-4 text-sm text-slate-600">
        Pick a case to view the image and submit your interpretation.
      </p>

      {isLoading && <p className="text-slate-500">Loading cases…</p>}
      {isError && (
        <p className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {error instanceof Error ? error.message : "Failed to load cases."}
        </p>
      )}

      {!isLoading && !isError && total > 0 && (
        <p className="mb-3 text-xs text-slate-500">
          Showing {cases.length} of {total} case{total === 1 ? "" : "s"}.
        </p>
      )}

      {!isLoading && !isError && total === 0 && (
        <p className="text-slate-500">No cases available yet.</p>
      )}

      {cases.length > 0 && (
        <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {cases.map((c) => {
            const demo = demographicsLine(c);
            return (
              <li key={c.id} className="card overflow-hidden">
                <Link to={`/cases/${c.id}`} className="block">
                  <div className="aspect-square w-full bg-slate-900">
                    <img
                      src={c.imageUrl}
                      alt={`Chest radiograph, ${caseTitle(c.id)}`}
                      loading="lazy"
                      className="h-full w-full object-cover"
                    />
                  </div>
                  <div className="p-3">
                    <div className="text-sm font-semibold text-slate-900">
                      {caseTitle(c.id)}
                    </div>
                    {demo !== null && (
                      <div className="mt-0.5 text-xs text-slate-500">{demo}</div>
                    )}
                    {c.clinicalContext && (
                      <p className="mt-2 line-clamp-2 text-sm text-slate-600">
                        {c.clinicalContext}
                      </p>
                    )}
                  </div>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
