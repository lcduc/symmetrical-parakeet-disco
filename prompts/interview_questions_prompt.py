from models.candidate import CandidateProfile
from models.job import JobRequirements

FORBIDDEN_INVENTION_NOTICE = (
    "Use ONLY the candidate and job information listed below. Do NOT reference, "
    "assume, or invent any experience, credential, employer, or requirement that "
    "is not explicitly listed."
)


def build_interview_prompt(candidate: CandidateProfile, job: JobRequirements) -> str:
    """Build a prompt asking the LLM for exactly 5 interview questions as a JSON list of strings.

    Q1-2 deep-dive the candidate's strongest relevant experience, Q3-4 probe
    explicit gaps between the candidate's experience/education and the job's
    requirements (naming the gap), and Q5 is a role-specific behavioral question.
    """
    return (
        "You are helping a recruiter prepare for a candidate interview. "
        "Generate exactly 5 interview questions for this candidate and job. "
        f"{FORBIDDEN_INVENTION_NOTICE}\n\n"
        "Requirements for the 5 questions:\n"
        "1-2. Deep-dive questions about the candidate's strongest experience "
        "relevant to this job.\n"
        "3-4. Questions that probe an explicit gap between the candidate's listed "
        "experience/education and the job's requirements or responsibilities. "
        "Each of these questions must name the specific gap being probed.\n"
        "5. One role-specific behavioral question relevant to this job's "
        "responsibilities.\n\n"
        f"Candidate experience: {candidate.experience_summary() or 'none listed'}\n"
        f"Candidate education: {candidate.education_summary() or 'none listed'}\n\n"
        f"Job title: {job.title}\n"
        f"Job requirements/responsibilities: {job.responsibilities_text or job.jd_text}\n"
        f"Job minimum years of experience: {job.min_years_experience}\n"
        f"Job required education level: {job.required_education_level or 'none'}\n\n"
        "Respond with ONLY a JSON list of exactly 5 strings, no preamble, no "
        'headings, no markdown code fences. Example format: ["question one", '
        '"question two", "question three", "question four", "question five"]'
    )
