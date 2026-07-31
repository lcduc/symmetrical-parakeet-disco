import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from api.security import enforce_rate_limit, require_api_key
from config.settings import settings
from db import async_session_factory
from models.candidate import CandidateProfile
from models.job import JobRequirements
from models.match import MatchResult
from models.orm_models import Candidate, Job, Match
from services.embedding_service import EmbeddingService
from services.llm_adapter import build_llm_adapter
from services.matching_service import MatchingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/matches", tags=["matches"], dependencies=[Depends(require_api_key)])


class MatchResponse(BaseModel):
    candidate_id: str
    candidate_name: str
    job_id: str
    passed_hard_filters: bool
    hard_filter_notes: str
    soft_fit_score: float
    overall_score: float
    explanation: str
    experience_flag: str


@router.post("/run", response_model=list[MatchResponse], dependencies=[Depends(enforce_rate_limit)])
async def run_matches(job_id: uuid.UUID, request: Request) -> list[MatchResponse]:
    """Match every successfully-processed candidate against a job, persist, and return ranked results."""
    async with async_session_factory() as session:
        job = await session.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        candidates = (
            (await session.execute(select(Candidate).where(Candidate.extraction_status == "ok")))
            .scalars()
            .all()
        )

    job_requirements = JobRequirements(
        job_id=str(job.id),
        title=job.title,
        jd_text=job.jd_text,
        **job.requirements_json,
    )

    embeddings = EmbeddingService(model=request.app.state.embedding_model, session_factory=async_session_factory)
    matcher = MatchingService(
        embeddings=embeddings, llm=build_llm_adapter(), hard_filter_soft_cap=settings.hard_filter_soft_cap
    )

    candidate_name_by_id = {str(candidate.id): candidate.full_name for candidate in candidates}

    # Matching (including the per-candidate LLM explanation call) runs with no
    # DB session open, so a slow batch doesn't tie up a connection-pool slot
    # for the duration of every candidate's LLM round-trip.
    results: list[MatchResult] = []
    for candidate in candidates:
        profile = CandidateProfile.model_validate({**candidate.structured_json, "candidate_id": str(candidate.id)})
        try:
            result = await matcher.match(profile, job_requirements)
        except Exception:
            logger.exception("Matching failed for candidate %s against job %s", candidate.id, job.id)
            continue
        results.append(result)

    async with async_session_factory() as session:
        for result in results:
            stmt = insert(Match).values(
                candidate_id=uuid.UUID(result.candidate_id),
                job_id=job.id,
                passed_hard_filters=result.passed_hard_filters,
                hard_filter_notes=result.hard_filter_notes,
                soft_fit_score=result.soft_fit_score,
                overall_score=result.overall_score,
                explanation=result.explanation,
                experience_flag=result.experience_flag,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[Match.candidate_id, Match.job_id],
                set_={
                    "passed_hard_filters": stmt.excluded.passed_hard_filters,
                    "hard_filter_notes": stmt.excluded.hard_filter_notes,
                    "soft_fit_score": stmt.excluded.soft_fit_score,
                    "overall_score": stmt.excluded.overall_score,
                    "explanation": stmt.excluded.explanation,
                    "experience_flag": stmt.excluded.experience_flag,
                },
            )
            await session.execute(stmt)
        await session.commit()

    results.sort(key=lambda result: result.overall_score, reverse=True)
    return [
        MatchResponse(**result.model_dump(), candidate_name=candidate_name_by_id.get(result.candidate_id, ""))
        for result in results
    ]
