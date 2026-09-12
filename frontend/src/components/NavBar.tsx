import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "@/auth/AuthContext";
import type { ReactNode } from "react";

export function NavBar(): ReactNode {
  const { user, logout, hasRole } = useAuth();
  const navigate = useNavigate();

  const handleLogout = (): void => {
    logout();
    navigate("/login");
  };

  return (
    <header className="border-b border-slate-200 bg-white">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <div className="flex items-center gap-6">
          <Link to="/cases" className="text-lg font-semibold text-brand-600">
            CXT Companion
          </Link>
          <Link to="/cases" className="text-sm text-slate-700 hover:text-brand-600">
            Cases
          </Link>
          {hasRole("EDUCATOR", "ADMIN") && (
            <Link
              to="/reviews"
              className="text-sm text-slate-700 hover:text-brand-600"
            >
              Review Queue
            </Link>
          )}
        </div>
        <div className="flex items-center gap-3 text-sm">
          {user !== null ? (
            <>
              <span className="text-slate-600">
                {user.email}{" "}
                <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-700">
                  {user.role}
                </span>
              </span>
              <button
                type="button"
                className="btn-ghost"
                onClick={handleLogout}
              >
                Log out
              </button>
            </>
          ) : (
            <Link to="/login" className="btn-primary">
              Log in
            </Link>
          )}
        </div>
      </nav>
    </header>
  );
}
