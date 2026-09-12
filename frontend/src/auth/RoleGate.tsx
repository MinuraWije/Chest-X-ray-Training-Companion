import { Navigate } from "react-router-dom";
import { useAuth } from "@/auth/AuthContext";
import type { ReactNode } from "react";
import type { Role } from "@/types/api";

interface RoleGateProps {
  allow: Role[];
  children: ReactNode;
  redirectTo?: string;
}

export function RoleGate({
  allow,
  children,
  redirectTo = "/cases",
}: RoleGateProps): ReactNode {
  const { user, status } = useAuth();
  if (status === "loading") {
    return (
      <div className="flex h-full items-center justify-center text-slate-500">
        Loading…
      </div>
    );
  }
  if (user === null || !allow.includes(user.role)) {
    return <Navigate to={redirectTo} replace />;
  }
  return <>{children}</>;
}
