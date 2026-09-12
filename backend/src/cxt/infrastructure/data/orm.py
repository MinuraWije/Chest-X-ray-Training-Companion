"""SQLAlchemy ORM base + models for users, attempts, and reviews.

These models are persistence-only. Domain entities (`User`, `StudentAttempt`,
`EducatorReview`) are the source of truth for the application layer; the
ORM rows are translated in the `Sqlite*Repository` adapters.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class UserRow(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(
            "role IN ('STUDENT', 'EDUCATOR', 'ADMIN')",
            name="ck_users_role",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AttemptRow(Base):
    __tablename__ = "attempts"
    __table_args__ = (
        Index("ix_attempts_student_id", "student_id"),
        Index("ix_attempts_submitted_at", "submitted_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    student_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    free_text: Mapped[str] = mapped_column(Text, nullable=False)
    feedback_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ReviewRow(Base):
    __tablename__ = "reviews"
    __table_args__ = (Index("ix_reviews_attempt_id", "attempt_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    attempt_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("attempts.id", ondelete="CASCADE"), nullable=False
    )
    educator_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    scores_json: Mapped[str] = mapped_column(Text, nullable=False)
    comment: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


def to_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row_dict.items() if v is not None}
