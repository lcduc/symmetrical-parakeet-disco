from models.candidate import CandidateProfile
from models.job import JobRequirements

FORBIDDEN_INVENTION_NOTICE = (
    "Use ONLY the candidate experience and job requirements listed below. Do "
    "NOT invent, infer, or assume any experience, credential, or requirement "
    "that is not explicitly listed."
)


def build_match_explanation_prompt(
    candidate: CandidateProfile,
    job: JobRequirements,
    passed: bool,
    notes: str,
    soft_fit: float,
    experience_flag: str = "",
) -> str:
    """Build a prompt asking the LLM for a 2-3 sentence explanation of a candidate-job match.

    The explanation should center on how closely the candidate's experience
    matches the job's stated requirements/responsibilities, since that
    similarity score is what drives the overall match percentage.
    """
    return (
        "Write a 2-3 sentence explanation of how well this candidate's experience "
        "matches this job's requirements, for a recruiter to read. "
        f"{FORBIDDEN_INVENTION_NOTICE}\n\n"
        f"Candidate experience: {candidate.experience_summary() or 'none listed'}\n"
        f"Candidate education: {candidate.education_summary() or 'none listed'}\n\n"
        f"Job title: {job.title}\n"
        f"Job requirements/responsibilities: {job.responsibilities_text or job.jd_text}\n"
        f"Job minimum years of experience: {job.min_years_experience}\n"
        f"Job required education level: {job.required_education_level or 'none'}\n\n"
        f"Hard requirements met: {passed}\n"
        f"Hard requirement notes: {notes}\n"
        f"Experience-years flag: {experience_flag or 'none'}\n"
        f"Experience-similarity match score (0-1): {soft_fit:.2f}\n\n"
        "Respond with the explanation only, no preamble or headings."
    )
