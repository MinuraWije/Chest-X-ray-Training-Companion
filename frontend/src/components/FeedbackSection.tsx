import type { ReactNode } from "react";
import type { Feedback } from "@/types/api";

interface FeedbackSectionProps {
  heading: string;
  body: string;
}

function Block({ heading, body }: FeedbackSectionProps): ReactNode {
  return (
    <section className="mb-3 last:mb-0">
      <h3 className="text-sm font-semibold text-slate-900">{heading}</h3>
      <p className="mt-1 whitespace-pre-wrap text-sm text-slate-700">{body}</p>
    </section>
  );
}

interface FeedbackViewProps {
  feedback: Feedback;
}

export function FeedbackSection({ feedback }: FeedbackViewProps): ReactNode {
  const { primaryFinding, differentialFindings, keyObservations, teachingPoint, sources } =
    feedback;

  return (
    <div className="card p-4">
      <h2 className="mb-3 text-lg font-semibold text-slate-900">Model feedback</h2>

      <div className="mb-4 rounded-md border border-brand-100 bg-brand-50 p-3">
        <div className="text-xs uppercase tracking-wide text-brand-700">
          Primary finding
        </div>
        <div className="mt-1 text-sm font-semibold text-slate-900">
          {primaryFinding.label}
        </div>
        <div className="text-xs text-slate-600">
          Confidence: {(primaryFinding.confidence * 100).toFixed(0)}%
        </div>
        <p className="mt-2 whitespace-pre-wrap text-sm text-slate-700">
          {primaryFinding.rationale}
        </p>
      </div>

      {differentialFindings.length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-slate-900">Differential</h3>
          <ul className="mt-1 space-y-1 text-sm text-slate-700">
            {differentialFindings.map((f) => (
              <li key={`${f.category}-${f.label}`} className="flex justify-between">
                <span>{f.label}</span>
                <span className="tabular-nums text-slate-500">
                  {(f.confidence * 100).toFixed(0)}%
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <Block heading="Key observations" body={keyObservations} />
      <Block heading="Teaching point" body={teachingPoint} />

      {sources.length > 0 && (
        <div className="mt-3 border-t border-slate-200 pt-3">
          <h3 className="text-xs uppercase tracking-wide text-slate-500">
            Sources
          </h3>
          <ul className="mt-1 space-y-1 text-xs text-slate-600">
            {sources.map((s) => (
              <li key={`${s.source}-${s.score}`}>
                <span className="font-medium text-slate-700">{s.source}</span>{" "}
                (score {(s.score * 100).toFixed(0)}%) — {s.text}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
