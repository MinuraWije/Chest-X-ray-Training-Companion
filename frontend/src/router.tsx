import { Navigate, Route, Routes } from "react-router-dom";
import type { ReactNode } from "react";
import { LoginPage } from "@/pages/LoginPage";
import { RegisterPage } from "@/pages/RegisterPage";
import { CaseListPage } from "@/pages/CaseListPage";
import { CasePage } from "@/pages/CasePage";
import { FeedbackPage } from "@/pages/FeedbackPage";
import { ReviewQueuePage } from "@/pages/ReviewQueuePage";
import { ReviewPage } from "@/pages/ReviewPage";
import { ProtectedRoute } from "@/auth/ProtectedRoute";
import { RoleGate } from "@/auth/RoleGate";
import { NavBar } from "@/components/NavBar";

export function AppRouter(): ReactNode {
  return (
    <div className="flex min-h-full flex-col">
      <NavBar />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Navigate to="/cases" replace />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          <Route
            path="/cases"
            element={
              <ProtectedRoute>
                <CaseListPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/cases/:caseId"
            element={
              <ProtectedRoute>
                <CasePage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/attempts/:attemptId/feedback"
            element={
              <ProtectedRoute>
                <FeedbackPage />
              </ProtectedRoute>
            }
          />

          <Route
            path="/reviews"
            element={
              <ProtectedRoute>
                <RoleGate allow={["EDUCATOR", "ADMIN"]}>
                  <ReviewQueuePage />
                </RoleGate>
              </ProtectedRoute>
            }
          />
          <Route
            path="/reviews/:attemptId"
            element={
              <ProtectedRoute>
                <RoleGate allow={["EDUCATOR", "ADMIN"]}>
                  <ReviewPage />
                </RoleGate>
              </ProtectedRoute>
            }
          />

          <Route path="*" element={<Navigate to="/cases" replace />} />
        </Routes>
      </main>
    </div>
  );
}
