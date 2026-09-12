import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { configureApiClient, getStoredToken } from "@/api/client";
import * as authApi from "@/api/auth";
import type { AuthUser, Role } from "@/types/api";

export interface AuthContextValue {
  user: AuthUser | null;
  status: "loading" | "authenticated" | "unauthenticated";
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
  hasRole: (...roles: Role[]) => boolean;
}

const AuthContext = createContext<AuthContextValue | null>(null);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps): ReactNode {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [status, setStatus] = useState<"loading" | "authenticated" | "unauthenticated">(
    "loading"
  );

  useEffect(() => {
    configureApiClient({
      tokenProvider: () => getStoredToken(),
      unauthorizedHandler: () => {
        setUser(null);
        setStatus("unauthenticated");
        if (window.location.pathname !== "/login") {
          window.location.assign("/login");
        }
      },
    });

    const token = getStoredToken();
    if (token === null) {
      setStatus("unauthenticated");
      return;
    }

    let cancelled = false;
    authApi
      .me({ suppressErrorToast: true })
      .then((u) => {
        if (cancelled) return;
        setUser(u);
        setStatus("authenticated");
      })
      .catch(() => {
        if (cancelled) return;
        setUser(null);
        setStatus("unauthenticated");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(
    async (email: string, password: string): Promise<void> => {
      const u = await authApi.login({ email, password });
      setUser(u);
      setStatus("authenticated");
    },
    []
  );

  const register = useCallback(
    async (email: string, password: string): Promise<void> => {
      const u = await authApi.register({ email, password });
      setUser(u);
      setStatus("authenticated");
    },
    []
  );

  const logout = useCallback((): void => {
    authApi.logout();
    setUser(null);
    setStatus("unauthenticated");
  }, []);

  const hasRole = useCallback(
    (...roles: Role[]): boolean => {
      if (user === null) return false;
      return roles.includes(user.role);
    },
    [user]
  );

  const value = useMemo<AuthContextValue>(
    () => ({ user, status, login, register, logout, hasRole }),
    [user, status, login, register, logout, hasRole]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (ctx === null) {
    throw new Error("useAuth must be used inside <AuthProvider>.");
  }
  return ctx;
}
