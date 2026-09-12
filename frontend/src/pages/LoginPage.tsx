import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useState, type ReactNode } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "@/auth/AuthContext";
import { EducationalBanner } from "@/components/EducationalBanner";

const schema = z.object({
  email: z.string().email("Enter a valid email."),
  password: z.string().min(8, "Password must be at least 8 characters."),
});

type LoginValues = z.infer<typeof schema>;

export function LoginPage(): ReactNode {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors, isValid },
  } = useForm<LoginValues>({
    resolver: zodResolver(schema),
    mode: "onChange",
    defaultValues: { email: "", password: "" },
  });

  const redirectTo = (() => {
    const state = location.state as { from?: string } | null;
    return state?.from ?? "/cases";
  })();

  return (
    <div className="mx-auto max-w-md p-6">
      <EducationalBanner />
      <h1 className="mb-1 text-2xl font-semibold">Log in</h1>
      <p className="mb-4 text-sm text-slate-600">
        Sign in with your student or educator account.
      </p>
      <form
        className="card p-4"
        onSubmit={handleSubmit(async (values) => {
          setSubmitting(true);
          setError(null);
          try {
            await login(values.email, values.password);
            navigate(redirectTo, { replace: true });
          } catch (e) {
            setError(e instanceof Error ? e.message : "Login failed.");
          } finally {
            setSubmitting(false);
          }
        })}
      >
        <label className="label" htmlFor="email">Email</label>
        <input id="email" type="email" autoComplete="email" className="input" {...register("email")} />
        {errors.email !== undefined && (
          <p className="mt-1 text-sm text-red-600">{errors.email.message}</p>
        )}

        <label className="label mt-3" htmlFor="password">Password</label>
        <input id="password" type="password" autoComplete="current-password" className="input" {...register("password")} />
        {errors.password !== undefined && (
          <p className="mt-1 text-sm text-red-600">{errors.password.message}</p>
        )}

        {error !== null && (
          <p className="mt-3 rounded-md border border-red-200 bg-red-50 p-2 text-sm text-red-700">
            {error}
          </p>
        )}

        <button
          type="submit"
          className="btn-primary mt-4 w-full"
          disabled={!isValid || submitting}
        >
          {submitting ? "Logging in…" : "Log in"}
        </button>
      </form>
      <p className="mt-3 text-sm text-slate-600">
        New student? <a href="/register" className="text-brand-600 hover:underline">Create an account</a>.
      </p>
    </div>
  );
}
