import { useState, type ReactNode } from "react";
import type {
  AnswerComparison as Comparison,
  ComparisonItem,
  ComparisonVerdict,
} from "@/types/api";

interface VerdictStyle {
  icon: string;
  /** Classes for a highlighted span in the student's text. */
  mark: string;
  /** Short status for the checklist row. */
  status: string;
}

// Colour is never the only signal: every verdict also carries an icon and
// a word, so the result reads the same without colour vision.
const STYLES: Record<ComparisonVerdict, VerdictStyle> = {
  correct: {
    icon: "✓",
    mark: "bg-green-100 text-green-900 ring-green-300",
    status: "Identified",
  },
  also_present: {
    icon: "✓",
    mark: "bg-green-100 text-green-900 ring-green-300",
    status: "Identified",
  },
  not_in_record: {
    icon: "⚠",
    mark: "bg-amber-100 text-amber-900 ring-amber-300",
    status: "Not in the case record",
  },
  wrongly_excluded: {
    icon: "✗",
    mark: "bg-red-100 text-red-900 ring-red-300",
    status: "You ruled this out",
  },
  correctly_excluded: {
    icon: "○",
    mark: "bg-slate-100 text-slate-800 ring-slate-300",
    status: "Correctly ruled out",
  },
  missed: {
    icon: "✗",
    mark: "",
    status: "Not mentioned",
  },
};

function explain(item: ComparisonItem): string {
  const name = item.label.toLowerCase();
  switch (item.verdict) {
    case "correct":
      return `You named ${name}, the confirmed finding on this film.`;
    case "also_present":
      return `You named ${name}, which is also confirmed on this film.`;
    case "not_in_record":
      return item.category === "no_finding"
        ? "You described this film as normal, but the case record lists a finding."
        : `The case record does not list ${name} for this film.`;
    case "wrongly_excluded":
      return `Your answer reads as ruling out ${name}, but it is confirmed on this film.`;
    case "correctly_excluded":
      return `You ruled out ${name}, and the case record agrees.`;
    case "missed":
      return item.category === "no_finding"
        ? "This film is recorded as normal, but your answer reports an abnormality."
        : `${item.label} is confirmed on this film, but your answer does not mention it.`;
  }
}

interface Segment {
  text: string;
  /** Index into `comparison.items`, or null for plain text. */
  item: number | null;
}

/**
 * Cut the answer into plain and highlighted runs.
 *
 * Spans come from the server and index the same `freeText` it returned, but
 * the component still refuses to trust them blindly: a span past the end of
 * the text, or one overlapping an earlier span, is dropped rather than
 * allowed to garble the student's own words.
 */
function segment(text: string, items: ComparisonItem[]): Segment[] {
  const spans = items
    .flatMap((item, index) => item.spans.map((span) => ({ ...span, item: index })))
    .filter((s) => s.start >= 0 && s.end <= text.length && s.start < s.end)
    .sort((a, b) => a.start - b.start);

  const segments: Segment[] = [];
  let cursor = 0;
  for (const span of spans) {
    if (span.start < cursor) continue;
    if (span.start > cursor) {
      segments.push({ text: text.slice(cursor, span.start), item: null });
    }
    segments.push({ text: text.slice(span.start, span.end), item: span.item });
    cursor = span.end;
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor), item: null });
  return segments;
}

const CONFIRMED: ReadonlySet<ComparisonVerdict> = new Set([
  "correct",
  "also_present",
  "missed",
  "wrongly_excluded",
]);

interface AnswerComparisonProps {
  freeText: string;
  comparison: Comparison;
  /** Id of the element holding the model's key observations, if shown. */
  whereToLookId?: string | undefined;
}

export function AnswerComparison({
  freeText,
  comparison,
  whereToLookId,
}: AnswerComparisonProps): ReactNode {
  const [active, setActive] = useState<number | null>(null);
  const { items, matchedAnything } = comparison;
  const confirmed = items.filter((i) => CONFIRMED.has(i.verdict));
  const others = items.filter((i) => !CONFIRMED.has(i.verdict));
  const activeItem = active !== null ? items[active] : undefined;

  return (
    <div className="card p-4">
      <h2 className="text-lg font-semibold text-slate-900">Your answer, compared</h2>

      <p className="mt-2 whitespace-pre-wrap text-sm leading-7 text-slate-700">
        {segment(freeText, items).map((seg, index) => {
          if (seg.item === null) return <span key={index}>{seg.text}</span>;
          const item = items[seg.item];
          if (item === undefined) return <span key={index}>{seg.text}</span>;
          const style = STYLES[item.verdict];
          const itemIndex = seg.item;
          return (
            <button
              key={index}
              type="button"
              title={explain(item)}
              aria-pressed={active === itemIndex}
              onClick={() => setActive(active === itemIndex ? null : itemIndex)}
              className={`rounded px-0.5 ring-1 ${style.mark} ${
                active === itemIndex ? "ring-2" : ""
              } focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500`}
            >
              <span aria-hidden="true" className="mr-0.5 text-xs">
                {style.icon}
              </span>
              {seg.text}
            </button>
          );
        })}
      </p>

      <p aria-live="polite" className="mt-2 min-h-[1.25rem] text-sm text-slate-800">
        {activeItem !== undefined
          ? explain(activeItem)
          : matchedAnything
            ? "Select a highlighted phrase to see how it compares with the case."
            : ""}
      </p>

      {!matchedAnything ? (
        <p className="mt-2 rounded-md border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
          We couldn&apos;t match any finding names in your answer. Compare it with
          the feedback instead.
        </p>
      ) : (
        <>
          <ChecklistGroup
            heading="On this film"
            items={confirmed}
            whereToLookId={whereToLookId}
          />
          {others.length > 0 && (
            <ChecklistGroup heading="Also in your answer" items={others} />
          )}
        </>
      )}

      <p className="mt-3 text-xs text-slate-500">
        Matched by keyword against the case record, not by reading for meaning.
        Misspellings and unusual phrasing may not be recognised, and the record
        does not say which side a finding is on.
      </p>
    </div>
  );
}

interface ChecklistGroupProps {
  heading: string;
  items: ComparisonItem[];
  whereToLookId?: string | undefined;
}

function ChecklistGroup({ heading, items, whereToLookId }: ChecklistGroupProps): ReactNode {
  return (
    <div className="mt-3">
      <h3 className="text-xs uppercase tracking-wide text-slate-500">{heading}</h3>
      <ul className="mt-1 space-y-1">
        {items.map((item) => {
          const style = STYLES[item.verdict];
          const needsHelp = item.verdict === "missed" || item.verdict === "wrongly_excluded";
          return (
            <li
              key={`${item.category}-${item.verdict}`}
              className="flex flex-wrap items-baseline gap-x-2 text-sm"
            >
              <span aria-hidden="true" className="w-4 text-center">
                {style.icon}
              </span>
              <span className="font-medium text-slate-900">{item.label}</span>
              <span className="text-slate-500">{style.status}</span>
              {needsHelp && whereToLookId !== undefined && (
                <a
                  href={`#${whereToLookId}`}
                  className="text-brand-700 underline hover:text-brand-600"
                >
                  See where to look
                </a>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
