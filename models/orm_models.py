import uuid
from datetime import datetime

from sqlalchemy import ARRAY, Float, ForeignKey, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name: Mapped[str] = mapped_column(default="")
    email: Mapped[str] = mapped_column(default="")
    phone: Mapped[str | None] = mapped_column(default=None)
    raw_file_path: Mapped[str]
    structured_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    extraction_status: Mapped[str] = mapped_column(default="pending", index=True)
    extraction_attempts: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str]
    jd_text: Mapped[str]
    requirements_json: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("candidate_id", "job_id"),
        Index("ix_matches_job_id", "job_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("candidates.id"))
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"))
    passed_hard_filters: Mapped[bool]
    hard_filter_notes: Mapped[str]
    soft_fit_score: Mapped[float]
    overall_score: Mapped[float]
    explanation: Mapped[str]
    experience_flag: Mapped[str] = mapped_column(default="")
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class InterviewQuestion(Base):
    __tablename__ = "interview_questions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("candidates.id"))
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"))
    questions_json: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class EmbeddingCache(Base):
    __tablename__ = "embedding_cache"

    cache_key: Mapped[str] = mapped_column(primary_key=True)
    vector: Mapped[list[float]] = mapped_column(ARRAY(Float))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
