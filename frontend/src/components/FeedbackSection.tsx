import type { ReactNode } from "react";
import type { Feedback, RetrievedPassage } from "@/types/api";

/** Anchor for the answer comparison's "See where to look" links. */
export const KEY_OBSERVATIONS_ID = "key-observations";

interface FeedbackSectionProps {
  heading: string;
  body: string;
  id?: string;
}

function Block({ heading, body, id }: FeedbackSectionProps): ReactNode {
  return (
    <section id={id} className="mb-3 scroll-mt-4 last:mb-0">
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

      <Block heading="Key observations" body={keyObservations} id={KEY_OBSERVATIONS_ID} />
      <Block heading="Teaching point" body={teachingPoint} />

      {sources.length > 0 && <SourceList sources={sources} />}
    </div>
  );
}

/**
 * The evidence panel — an in-app reading card per passage.
 *
 * Students stay in the app: each passage renders its own title and plain
 * teaching text, so nobody has to make sense of a raw corpus file that holds
 * many chunks plus build metadata. The repository URL is provenance for the
 * evaluation, not a reading destination, so it is deliberately not linked
 * from the student view.
 *
 * No score is shown. A percentage next to a citation implies a reliability the
 * number does not carry — BM25 rank is not a probability that the passage is
 * correct about this patient — and in a tool medical students are meant to
 * calibrate against, an authoritative-looking "87%" beside a keyword match is
 * worse than no number at all. The position in the list already conveys rank,
 * and the raw score is kept in `Feedback.raw.retrieval` for the evaluation.
 *
 * Passages are collapsed by default: a corpus chunk runs to 700 characters and
 * three of them would bury the teaching point they are meant to support.
 * A curated "Learn more" link is shown only when the chunk carries a verified
 * teaching article (`readMoreUrl`); otherwise no link is rendered at all.
 */
function SourceList({ sources }: { sources: RetrievedPassage[] }): ReactNode {
  return (
    <div className="mt-3 border-t border-slate-200 pt-3">
      <h3 className="text-xs uppercase tracking-wide text-slate-500">
        Reference passages
      </h3>
      <p className="mt-1 text-xs text-slate-500">
        Teaching notes supplied to the model. These describe findings in
        general and were not written about this patient.
      </p>
      <ul className="mt-2 space-y-2 text-xs text-slate-600">
        {sources.map((s, index) => (
          // Keyed on the chunk id, not `source`: one document yields many
          // chunks sharing a source string, which produced duplicate keys.
          <li key={s.id || `passage-${index}`}>
            <details className="group rounded-md border border-slate-200 bg-slate-50 p-2">
              <summary className="cursor-pointer list-none font-medium text-slate-700 hover:text-brand-700">
                <span className="mr-1 tabular-nums text-slate-400">
                  [S{index + 1}]
                </span>
                {s.source}
                <span className="ml-1 font-normal text-slate-400 group-open:hidden">
                  — read
                </span>
              </summary>
              <p className="mt-1 whitespace-pre-wrap pl-6 text-slate-600">{s.text}</p>
              <p className="mt-1 pl-6 text-[11px] text-slate-400">
                Teaching note · CC BY-NC-SA 3.0
              </p>
              {s.readMoreUrl && (
                <a
                  className="mt-1 inline-block pl-6 text-brand-700 underline hover:text-brand-800"
                  href={s.readMoreUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Learn more{s.readMoreLabel ? `: ${s.readMoreLabel}` : ""}
                </a>
              )}
            </details>
          </li>
        ))}
      </ul>
    </div>
  );
}
