import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import type { ReactNode } from "react";

const schema = z.object({
  freeText: z
    .string()
    .min(20, "Please describe at least 20 characters of findings.")
    .max(2000, "Please keep your description under 2000 characters."),
});

export type AttemptFormValues = z.infer<typeof schema>;

interface AttemptFormProps {
  onSubmit: (values: AttemptFormValues) => Promise<void> | void;
  submitting: boolean;
}

export function AttemptForm({ onSubmit, submitting }: AttemptFormProps): ReactNode {
  const {
    register,
    handleSubmit,
    formState: { errors, isValid },
  } = useForm<AttemptFormValues>({
    resolver: zodResolver(schema),
    mode: "onChange",
    defaultValues: { freeText: "" },
  });

  return (
    <form
      className="card p-4"
      onSubmit={handleSubmit(async (values) => {
        await onSubmit(values);
      })}
    >
      <h2 className="mb-2 text-lg font-semibold text-slate-900">Your interpretation</h2>
      <p className="mb-3 text-sm text-slate-600">
        Describe what you see in the image and your differential. Mention any
        key features (laterality, distribution, associated findings) that
        informed your top diagnosis.
      </p>
      <label className="label" htmlFor="freeText">
        Free-text report
      </label>
      <textarea
        id="freeText"
        className="input min-h-[160px]"
        placeholder="E.g. There is a small left apical pneumothorax…"
        {...register("freeText")}
      />
      {errors.freeText !== undefined && (
        <p className="mt-1 text-sm text-red-600">{errors.freeText.message}</p>
      )}
      <div className="mt-3 flex justify-end">
        <button
          type="submit"
          className="btn-primary"
          disabled={!isValid || submitting}
        >
          {submitting ? "Submitting…" : "Submit & request feedback"}
        </button>
      </div>
    </form>
  );
}
