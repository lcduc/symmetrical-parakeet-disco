import asyncio
import logging

from pydantic import ValidationError

from config.settings import settings
from models.candidate import CandidateProfile
from prompts.extraction_prompt import build_correction_prompt, build_extraction_prompt
from utils.education_level import infer_education_level

logger = logging.getLogger(__name__)

RETRY_BACKOFF_BASE_SECONDS = 2
RETRY_BACKOFF_CAP_SECONDS = 30


class ExtractionService:
    """Extracts a CandidateProfile from raw resume text via an LLM, with self-heal retries."""

    def __init__(self, llm, max_retries: int | None = None):
        self._llm = llm
        self._max_retries = max_retries if max_retries is not None else settings.max_extraction_retries

    async def extract(self, resume_text: str) -> tuple[CandidateProfile | None, int]:
        """Attempt to extract a CandidateProfile, retrying on validation failure or transient LLM error.

        Returns (profile, attempts_used). Never raises: if every retry is
        exhausted, returns (None, max_retries).
        """
        prompt = build_extraction_prompt(resume_text)

        for attempt in range(1, self._max_retries + 1):
            logger.info("Extraction attempt %d/%d", attempt, self._max_retries)
            try:
                raw = await self._llm.complete_structured(prompt)
            except Exception:
                logger.exception(
                    "Extraction attempt %d/%d hit a transient LLM error; retrying", attempt, self._max_retries
                )
                if attempt < self._max_retries:
                    backoff = min(RETRY_BACKOFF_BASE_SECONDS**attempt, RETRY_BACKOFF_CAP_SECONDS)
                    await asyncio.sleep(backoff)
                continue

            try:
                profile = CandidateProfile.model_validate_json(raw)
                self._fill_missing_education_levels(profile)
                return profile, attempt
            except ValidationError as error:
                logger.info(
                    "Extraction attempt %d/%d failed validation: %s",
                    attempt,
                    self._max_retries,
                    error,
                )
                prompt = build_correction_prompt(resume_text, error)

        logger.error("Extraction exhausted all %d retries", self._max_retries)
        return None, self._max_retries

    @staticmethod
    def _fill_missing_education_levels(profile: CandidateProfile) -> None:
        """Infer each education entry's rank level from its degree text when the LLM left it null.

        MatchingService's hard filters compare against `level`, so a
        populated `degree` ("BSc", "PhD", ...) with a null `level` would
        otherwise silently fail every education-based hard filter.
        """
        for entry in profile.education:
            if not entry.level:
                entry.level = infer_education_level(entry.degree)
