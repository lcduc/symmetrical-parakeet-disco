import asyncio
import logging
import subprocess
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError
from sqlalchemy import select

from api.security import enforce_rate_limit, require_api_key
from config.settings import settings
from db import async_session_factory
from models.candidate import CandidateProfile, EducationEntry, WorkExperience
from models.orm_models import Candidate
from services.extraction_service import ExtractionService
from services.ingestion_service import IngestionService
from services.llm_adapter import build_llm_adapter
from services.ocr_router_service import OCRRouterService
from services.template_service import TemplateService
from utils.pii_scrub import scrub_free_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/candidates", tags=["candidates"], dependencies=[Depends(require_api_key)])

UPLOAD_DIR = Path("data/uploads")
RENDERED_DIR = Path("data/rendered")
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx"}


class UploadResponse(BaseModel):
    candidate_id: str
    status: str


class CandidateResponse(BaseModel):
    candidate_id: str
    status: str
    profile: dict | None = None


class CandidatePatchRequest(BaseModel):
    """Partial CandidateProfile update, used by the Needs Review manual-correction form."""

    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    skills: list[str] | None = None
    work_history: list[WorkExperience] | None = None
    education: list[EducationEntry] | None = None
    certifications: list[str] | None = None
    years_experience_total: float | None = None


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=UploadResponse,
    dependencies=[Depends(enforce_rate_limit)],
)
async def upload_candidate(
    background_tasks: BackgroundTasks, request: Request, file: UploadFile = File(...)
) -> UploadResponse:
    """Save the uploaded resume, insert a pending Candidate row, and process it in the background."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}",
        )

    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length is not None and int(content_length) > max_size_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large.")

    content = await file.read()
    if len(content) > max_size_bytes:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    candidate_id = uuid.uuid4()
    dest_path = UPLOAD_DIR / f"{candidate_id}{suffix}"
    dest_path.write_bytes(content)

    async with async_session_factory() as session:
        session.add(Candidate(id=candidate_id, raw_file_path=str(dest_path), extraction_status="pending"))
        await session.commit()

    background_tasks.add_task(_process_candidate, candidate_id, dest_path, request.app.state.ocr_router_service)
    return UploadResponse(candidate_id=str(candidate_id), status="pending")


async def _process_candidate(candidate_id: uuid.UUID, file_path: Path, ocr_router: OCRRouterService) -> None:
    """Run ingestion + extraction for an uploaded resume and persist the result."""
    try:
        resume_text = await asyncio.to_thread(IngestionService(ocr_router).extract_text, file_path)
        llm = build_llm_adapter(model=settings.extraction_llm_model)
        profile, attempts = await ExtractionService(llm).extract(resume_text)
    except Exception:
        logger.exception("Processing failed for candidate %s", candidate_id)
        profile, attempts = None, settings.max_extraction_retries

    async with async_session_factory() as session:
        candidate = await session.get(Candidate, candidate_id)
        if candidate is None:
            logger.error("Candidate %s disappeared before processing completed", candidate_id)
            return

        candidate.extraction_attempts = attempts
        if profile is None:
            candidate.extraction_status = "needs_review"
        else:
            for entry in profile.work_history:
                entry.description = [scrub_free_text(bullet) for bullet in entry.description]
            candidate.full_name = profile.full_name
            candidate.email = profile.email
            candidate.phone = profile.phone
            candidate.structured_json = profile.model_dump(mode="json")
            candidate.extraction_status = "ok"
        await session.commit()


@router.get("", response_model=list[CandidateResponse])
async def list_candidates(status: str | None = None) -> list[CandidateResponse]:
    """List candidates, optionally filtered by extraction_status."""
    async with async_session_factory() as session:
        stmt = select(Candidate)
        if status is not None:
            stmt = stmt.where(Candidate.extraction_status == status)
        candidates = (await session.execute(stmt)).scalars().all()
    return [
        CandidateResponse(
            candidate_id=str(candidate.id),
            status=candidate.extraction_status,
            profile=candidate.structured_json or None,
        )
        for candidate in candidates
    ]


@router.get("/{candidate_id}", response_model=CandidateResponse)
async def get_candidate(candidate_id: uuid.UUID) -> CandidateResponse:
    async with async_session_factory() as session:
        candidate = await session.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return CandidateResponse(
        candidate_id=str(candidate.id),
        status=candidate.extraction_status,
        profile=candidate.structured_json or None,
    )


@router.get("/{candidate_id}/file")
async def download_candidate_file(candidate_id: uuid.UUID) -> FileResponse:
    """Return the originally uploaded resume file, for display alongside the review form."""
    async with async_session_factory() as session:
        candidate = await session.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")

    raw_file_path = Path(candidate.raw_file_path)
    if not raw_file_path.exists():
        raise HTTPException(status_code=404, detail="Original file is no longer available")
    return FileResponse(path=raw_file_path, filename=raw_file_path.name)


@router.patch("/{candidate_id}", response_model=CandidateResponse)
async def patch_candidate(candidate_id: uuid.UUID, body: CandidatePatchRequest) -> CandidateResponse:
    """Apply manual corrections to a candidate's profile and mark it as ok.

    Used by the Needs Review tab: merges the submitted fields into whatever
    partial structured_json already exists, validates the result as a full
    CandidateProfile, and persists it.
    """
    async with async_session_factory() as session:
        candidate = await session.get(Candidate, candidate_id)
        if candidate is None:
            raise HTTPException(status_code=404, detail="Candidate not found")

        if body.work_history is not None:
            for entry in body.work_history:
                entry.description = [scrub_free_text(bullet) for bullet in entry.description]

        merged = {**(candidate.structured_json or {}), **body.model_dump(exclude_unset=True)}
        merged["candidate_id"] = str(candidate.id)
        try:
            profile = CandidateProfile.model_validate(merged)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        candidate.full_name = profile.full_name
        candidate.email = profile.email
        candidate.phone = profile.phone
        candidate.structured_json = profile.model_dump(mode="json")
        candidate.extraction_status = "ok"
        await session.commit()

        return CandidateResponse(
            candidate_id=str(candidate.id),
            status=candidate.extraction_status,
            profile=candidate.structured_json,
        )


@router.post("/{candidate_id}/render-template")
async def render_candidate_template(candidate_id: uuid.UUID) -> FileResponse:
    """Render the candidate's structured profile into the company LaTeX resume template as a PDF."""
    async with async_session_factory() as session:
        candidate = await session.get(Candidate, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    if not candidate.structured_json:
        raise HTTPException(status_code=409, detail="Candidate has no extracted profile yet")

    profile = CandidateProfile.model_validate(candidate.structured_json)
    output_path = RENDERED_DIR / f"{candidate_id}.pdf"
    try:
        await asyncio.to_thread(TemplateService().render, profile, output_path)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.exception("Template rendering failed for candidate %s", candidate_id)
        raise HTTPException(status_code=500, detail="Failed to render resume template") from None

    return FileResponse(
        path=output_path,
        media_type="application/pdf",
        filename=f"{profile.full_name or candidate_id}.pdf",
    )
