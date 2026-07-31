from pydantic import BaseModel


class MatchResult(BaseModel):
    candidate_id: str
    job_id: str
    passed_hard_filters: bool
    hard_filter_notes: str
    soft_fit_score: float
    overall_score: float
    explanation: str
    experience_flag: str = ""
