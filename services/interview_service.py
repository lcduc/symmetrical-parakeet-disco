import json
import logging

from models.candidate import CandidateProfile
from models.job import JobRequirements
from prompts.interview_questions_prompt import build_interview_prompt
from services.llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)

EXPECTED_QUESTION_COUNT = 5


class InterviewService:
    """Generates candidate-specific interview questions via an LLM."""

    def __init__(self, llm: LLMAdapter):
        self._llm = llm

    async def generate_questions(self, candidate: CandidateProfile, job: JobRequirements) -> list[str]:
        """Ask the LLM for 5 interview questions and return them as a list of strings.

        Returns an empty list if the LLM response is not valid JSON. Logs a
        warning (without raising) if the parsed result isn't a 5-item list.
        """
        prompt = build_interview_prompt(candidate, job)
        response = await self._llm.complete_structured(prompt)

        try:
            questions = json.loads(response)
        except json.JSONDecodeError:
            logger.exception(
                "Failed to parse interview questions JSON for candidate %s / job %s",
                candidate.candidate_id,
                job.job_id,
            )
            return []

        if not isinstance(questions, list) or len(questions) != EXPECTED_QUESTION_COUNT:
            logger.warning(
                "Expected %d interview questions for candidate %s / job %s, got: %r",
                EXPECTED_QUESTION_COUNT,
                candidate.candidate_id,
                job.job_id,
                questions,
            )

        return questions if isinstance(questions, list) else []
