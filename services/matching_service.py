import hashlib
import logging

import numpy as np

from models.candidate import CandidateProfile
from models.job import JobRequirements
from models.match import MatchResult
from prompts.match_explanation_prompt import build_match_explanation_prompt

logger = logging.getLogger(__name__)

EDUCATION_RANK = {
    "high_school": 0,
    "associate": 1,
    "bachelor": 2,
    "master": 3,
    "phd": 4,
}

# A candidate with at least this fraction of the required years is not hard-filtered
# out, but is flagged for human review if they fall short of the exact requirement.
YEARS_EXPERIENCE_TOLERANCE_RATIO = 0.5


def _content_key(text: str) -> str:
    """Short content-derived cache-key fragment so an edited bullet/line gets a fresh embedding cache entry."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors; returns 0.0 if either has zero norm."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class MatchingService:
    """Scores candidates against job requirements via hard filters and experience-similarity fit.

    The embedding model and LLM adapter are expensive to construct, so they
    are built once at app startup and injected here rather than reloaded per call.
    """

    def __init__(self, embeddings, llm, hard_filter_soft_cap: float = 0.4):
        self._embeddings = embeddings
        self._llm = llm
        self._soft_cap = hard_filter_soft_cap

    def apply_hard_filters(self, candidate: CandidateProfile, job: JobRequirements) -> tuple[bool, str, str]:
        """Check candidate against job's non-negotiable requirements: years of experience and education.

        Years of experience accepts candidates down to
        YEARS_EXPERIENCE_TOLERANCE_RATIO of the requirement, but flags them for
        review rather than passing silently.

        Returns (passed, notes, experience_flag). notes lists every failing
        check or, if all checks pass, reads "Meets all hard requirements."
        experience_flag is "" unless the candidate was accepted below the
        exact years-of-experience requirement. Never raises.
        """
        failures: list[str] = []
        experience_flag = ""

        try:
            min_years = job.min_years_experience
            candidate_years = candidate.years_experience_total
            if min_years > 0:
                tolerance_floor = min_years * YEARS_EXPERIENCE_TOLERANCE_RATIO
                if candidate_years < tolerance_floor:
                    failures.append(
                        f"Requires at least {min_years} years of experience "
                        f"(minimum {tolerance_floor:.1f} accepted with tolerance); "
                        f"candidate has {candidate_years}."
                    )
                elif candidate_years < min_years:
                    experience_flag = (
                        f"Candidate has {candidate_years} years of experience vs "
                        f"{min_years} required — accepted within tolerance, flagged for review."
                    )

            if job.required_education_level:
                required_rank = EDUCATION_RANK.get(job.required_education_level.lower())
                candidate_rank = self._highest_education_rank(candidate)
                if required_rank is not None and (candidate_rank is None or candidate_rank < required_rank):
                    failures.append(f"Requires {job.required_education_level} education level or higher.")
        except Exception:
            logger.exception(
                "Error applying hard filters for candidate %s against job %s",
                candidate.candidate_id,
                job.job_id,
            )
            return False, "Error evaluating hard requirements.", ""

        if failures:
            return False, " ".join(failures), experience_flag
        return True, "Meets all hard requirements.", experience_flag

    @staticmethod
    def _highest_education_rank(candidate: CandidateProfile) -> int | None:
        """Return the highest EDUCATION_RANK value among candidate's education entries, or None if unranked."""
        ranks = [
            EDUCATION_RANK[entry.level.lower()]
            for entry in candidate.education
            if entry.level and entry.level.lower() in EDUCATION_RANK
        ]
        return max(ranks) if ranks else None

    @staticmethod
    def _candidate_experience_bullets(candidate: CandidateProfile) -> list[str]:
        """Flatten every work-history description bullet; fall back to the experience summary if none exist."""
        bullets = [bullet.strip() for entry in candidate.work_history for bullet in entry.description if bullet.strip()]
        if bullets:
            return bullets
        summary = candidate.experience_summary().strip()
        return [summary] if summary else []

    @staticmethod
    def _job_requirement_statements(job: JobRequirements) -> list[str]:
        """Split the job's responsibilities (or full JD text as a fallback) into individual requirement lines."""
        text = job.responsibilities_text.strip() or job.jd_text.strip()
        statements = [line.strip() for line in text.splitlines() if line.strip()]
        return statements if statements else ([text] if text else [])

    async def score_soft_fit(self, candidate: CandidateProfile, job: JobRequirements) -> float:
        """Embedding-similarity fit between every candidate experience bullet and every JD requirement line.

        For each JD requirement statement, finds the candidate experience
        bullet it best matches (max cosine similarity), then averages those
        best-match scores across all requirement statements. Returns 0.0 if
        the candidate has no experience bullets or the job has no requirement
        text to compare against.
        """
        candidate_bullets = self._candidate_experience_bullets(candidate)
        job_statements = self._job_requirement_statements(job)
        if not candidate_bullets or not job_statements:
            return 0.0

        candidate_vectors = [
            await self._embeddings.embed_cached(f"candidate:{candidate.candidate_id}:bullet:{_content_key(bullet)}", bullet)
            for bullet in candidate_bullets
        ]
        job_vectors = [
            await self._embeddings.embed_cached(f"job:{job.job_id}:requirement:{_content_key(statement)}", statement)
            for statement in job_statements
        ]

        best_matches = [
            max(_cosine(job_vector, candidate_vector) for candidate_vector in candidate_vectors)
            for job_vector in job_vectors
        ]
        return max(0.0, sum(best_matches) / len(best_matches))

    async def match(self, candidate: CandidateProfile, job: JobRequirements) -> MatchResult:
        """Run hard filters and experience-similarity scoring, returning a populated MatchResult."""
        passed, notes, experience_flag = self.apply_hard_filters(candidate, job)
        soft_fit = await self.score_soft_fit(candidate, job)
        overall_score = soft_fit if passed else min(soft_fit, self._soft_cap)
        explanation = await self._generate_explanation(candidate, job, passed, notes, soft_fit, experience_flag)
        return MatchResult(
            candidate_id=candidate.candidate_id,
            job_id=job.job_id,
            passed_hard_filters=passed,
            hard_filter_notes=notes,
            soft_fit_score=soft_fit,
            overall_score=overall_score,
            explanation=explanation,
            experience_flag=experience_flag,
        )

    async def _generate_explanation(
        self,
        candidate: CandidateProfile,
        job: JobRequirements,
        passed: bool,
        notes: str,
        soft_fit: float,
        experience_flag: str,
    ) -> str:
        """Generate a human-readable explanation of the match via the LLM.

        Falls back to a deterministic, notes-based explanation on LLM failure
        (rate limit, timeout, ...) so a transient provider error doesn't drop
        an otherwise-scored candidate from the ranked results entirely.
        """
        prompt = build_match_explanation_prompt(candidate, job, passed, notes, soft_fit, experience_flag)
        try:
            response = await self._llm.complete(prompt, max_tokens=200, temperature=0.2)
            return response.strip()
        except Exception:
            logger.exception(
                "Explanation generation failed for candidate %s against job %s; using fallback explanation",
                candidate.candidate_id,
                job.job_id,
            )
            fallback = f"{notes} Soft-fit score: {soft_fit * 100:.0f}%."
            if experience_flag:
                fallback += f" {experience_flag}"
            return fallback
