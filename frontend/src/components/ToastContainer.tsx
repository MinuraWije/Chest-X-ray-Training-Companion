import type { ReactNode } from "react";
import { useToast, type Toast, type ToastVariant } from "@/components/ToastProvider";

const VARIANT_STYLES: Record<ToastVariant, string> = {
  success: "border-emerald-300 bg-emerald-50 text-emerald-900",
  error: "border-red-300 bg-red-50 text-red-900",
  info: "border-sky-300 bg-sky-50 text-sky-900",
  warning: "border-amber-300 bg-amber-50 text-amber-900",
};

const VARIANT_ICONS: Record<ToastVariant, string> = {
  success: "✓",
  error: "✕",
  info: "i",
  warning: "!",
};

const VARIANT_ICON_BG: Record<ToastVariant, string> = {
  success: "bg-emerald-500 text-white",
  error: "bg-red-500 text-white",
  info: "bg-sky-500 text-white",
  warning: "bg-amber-500 text-white",
};

interface ToastItemProps {
  toast: Toast;
  onDismiss: (id: string) => void;
}

function ToastItem({ toast, onDismiss }: ToastItemProps): ReactNode {
  return (
    <div
      role={toast.variant === "error" ? "alert" : "status"}
      aria-live={toast.variant === "error" ? "assertive" : "polite"}
      className={`pointer-events-auto flex w-80 max-w-[calc(100vw-2rem)] items-start gap-3 rounded-md border p-3 shadow-md ${VARIANT_STYLES[toast.variant]}`}
    >
      <span
        aria-hidden="true"
        className={`mt-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-sm font-semibold ${VARIANT_ICON_BG[toast.variant]}`}
      >
        {VARIANT_ICONS[toast.variant]}
      </span>
      <div className="min-w-0 flex-1">
        <div className="text-sm font-semibold">{toast.title}</div>
        {toast.message !== undefined && (
          <p className="mt-1 whitespace-pre-wrap text-xs opacity-90">{toast.message}</p>
        )}
      </div>
      <button
        type="button"
        aria-label="Dismiss"
        className="ml-1 text-lg leading-none opacity-60 hover:opacity-100"
        onClick={() => onDismiss(toast.id)}
      >
        ×
      </button>
    </div>
  );
}

export function ToastContainer(): ReactNode {
  const { toasts, dismiss } = useToast();
  if (toasts.length === 0) {
    return null;
  }
  return (
    <div
      aria-label="Notifications"
      className="pointer-events-none fixed bottom-4 right-4 z-50 flex flex-col gap-2"
    >
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={dismiss} />
      ))}
    </div>
  );
}
