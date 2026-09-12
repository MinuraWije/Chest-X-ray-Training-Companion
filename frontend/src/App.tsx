import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useMemo, type ReactNode } from "react";
import { AuthProvider } from "@/auth/AuthContext";
import { AppRouter } from "@/router";
import { ToastProvider, useToast } from "@/components/ToastProvider";
import { ToastContainer } from "@/components/ToastContainer";
import { configureApiClient, getStoredToken } from "@/api/client";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
      refetchOnWindowFocus: false,
    },
  },
});

function AppShell(): ReactNode {
  const toast = useToast();
  const auth = useMemo(
    () => ({
      tokenProvider: () => getStoredToken(),
      unauthorizedHandler: () => {
        if (window.location.pathname !== "/login") {
          window.location.assign("/login");
        }
      },
      toastEmitter: toast.show,
    }),
    [toast.show],
  );
  configureApiClient({
    tokenProvider: auth.tokenProvider,
    unauthorizedHandler: auth.unauthorizedHandler,
    toastEmitter: auth.toastEmitter,
  });
  return (
    <BrowserRouter>
      <AppRouter />
      <ToastContainer />
    </BrowserRouter>
  );
}

function App(): ReactNode {
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <AuthProvider>
          <AppShell />
        </AuthProvider>
      </ToastProvider>
    </QueryClientProvider>
  );
}

export default App;
