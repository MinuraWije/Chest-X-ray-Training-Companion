import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

export type ToastVariant = "success" | "error" | "info" | "warning";

export interface ToastInput {
  variant: ToastVariant;
  title: string;
  message?: string | undefined;
  durationMs?: number | undefined;
}

export interface Toast extends Required<Pick<ToastInput, "variant" | "title">> {
  id: string;
  message: string | undefined;
  durationMs: number;
}

export interface ToastContextValue {
  toasts: readonly Toast[];
  show: (toast: ToastInput) => void;
  dismiss: (id: string) => void;
  success: (title: string, message?: string) => void;
  error: (title: string, message?: string) => void;
  info: (title: string, message?: string) => void;
  warning: (title: string, message?: string) => void;
}

const DEFAULT_DURATION_MS = 5_000;
const ERROR_DURATION_MS = 8_000;

const ToastContext = createContext<ToastContextValue | null>(null);

interface ToastProviderProps {
  children: ReactNode;
}

export function ToastProvider({ children }: ToastProviderProps): ReactNode {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const counterRef = useRef<number>(0);

  const dismiss = useCallback((id: string): void => {
    setToasts((current) => current.filter((t) => t.id !== id));
  }, []);

  const show = useCallback(
    (input: ToastInput): void => {
      counterRef.current += 1;
      const id = `t${Date.now()}-${counterRef.current}`;
      const duration =
        input.durationMs ??
        (input.variant === "error" ? ERROR_DURATION_MS : DEFAULT_DURATION_MS);
      const toast: Toast = {
        id,
        variant: input.variant,
        title: input.title,
        message: input.message,
        durationMs: duration,
      };
      setToasts((current) => [...current, toast]);
      if (duration > 0) {
        window.setTimeout(() => dismiss(id), duration);
      }
    },
    [dismiss],
  );

  const make = useCallback(
    (variant: ToastVariant) =>
      (title: string, message?: string): void => {
        show({ variant, title, message });
      },
    [show],
  );

  const value = useMemo<ToastContextValue>(
    () => ({
      toasts,
      show,
      dismiss,
      success: make("success"),
      error: make("error"),
      info: make("info"),
      warning: make("warning"),
    }),
    [toasts, show, dismiss, make],
  );

  return <ToastContext.Provider value={value}>{children}</ToastContext.Provider>;
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (ctx === null) {
    throw new Error("useToast must be used inside <ToastProvider>.");
  }
  return ctx;
}
