import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.security import require_api_key
from db import async_session_factory
from models.orm_models import Job

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(require_api_key)])


class JobCreateRequest(BaseModel):
    title: str
    jd_text: str
    min_years_experience: float = 0.0
    required_education_level: str | None = None
    responsibilities_text: str = ""


class JobCreateResponse(BaseModel):
    job_id: str


@router.post("", response_model=JobCreateResponse)
async def create_job(body: JobCreateRequest) -> JobCreateResponse:
    """Insert a Job row from a job description and its structured requirements."""
    requirements_json = {
        "min_years_experience": body.min_years_experience,
        "required_education_level": body.required_education_level,
        "responsibilities_text": body.responsibilities_text,
    }

    async with async_session_factory() as session:
        job = Job(title=body.title, jd_text=body.jd_text, requirements_json=requirements_json)
        session.add(job)
        await session.commit()
        await session.refresh(job)

    logger.info("Created job %s: %s", job.id, body.title)
    return JobCreateResponse(job_id=str(job.id))
