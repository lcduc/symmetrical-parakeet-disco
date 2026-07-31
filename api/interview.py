import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.security import require_api_key
from db import async_session_factory
from models.candidate import CandidateProfile
from models.job import JobRequirements
from models.orm_models import Candidate, InterviewQuestion, Job
from services.interview_service import InterviewService
from services.llm_adapter import build_llm_adapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/interview", tags=["interview"], dependencies=[Depends(require_api_key)])


class InterviewQuestionsResponse(BaseModel):
    candidate_id: str
    job_id: str
    questions: list[str]


@router.post("/generate", response_model=InterviewQuestionsResponse)
async def generate_interview_questions(candidate_id: uuid.UUID, job_id: uuid.UUID) -> InterviewQuestionsResponse:
    """Generate and persist interview questions for a candidate-job pair."""
    async with async_session_factory() as session:
        candidate = await session.get(Candidate, candidate_id)
        job = await session.get(Job, job_id)

    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not candidate.structured_json:
        raise HTTPException(status_code=409, detail="Candidate has no extracted profile yet")

    profile = CandidateProfile.model_validate({**candidate.structured_json, "candidate_id": str(candidate.id)})
    job_requirements = JobRequirements(
        job_id=str(job.id), title=job.title, jd_text=job.jd_text, **job.requirements_json
    )

    service = InterviewService(build_llm_adapter())
    questions = await service.generate_questions(profile, job_requirements)

    async with async_session_factory() as session:
        session.add(
            InterviewQuestion(
                candidate_id=candidate.id,
                job_id=job.id,
                questions_json={"questions": questions},
            )
        )
        await session.commit()

    return InterviewQuestionsResponse(candidate_id=str(candidate.id), job_id=str(job.id), questions=questions)
