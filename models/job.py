from pydantic import BaseModel


class JobRequirements(BaseModel):
    job_id: str = ""
    title: str
    jd_text: str
    min_years_experience: float = 0.0
    required_education_level: str | None
    responsibilities_text: str = ""
