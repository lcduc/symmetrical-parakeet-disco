from pydantic import BaseModel


class WorkExperience(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: str | None
    description: list[str] = []


class EducationEntry(BaseModel):
    """All fields are optional: a resume entry may be as sparse as a bare end_date."""

    institution: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    level: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    description: list[str] = []


class CandidateProfile(BaseModel):
    """Domain model for a candidate's resume-derived profile.

    Deliberately excludes date_of_birth, age, photo, nationality,
    marital_status, and gender: these are protected/sensitive attributes
    that must not factor into hard filters, soft scoring, or explanations
    produced downstream. This exclusion is intentional and permanent —
    do not add these fields back even if a source resume contains them.
    """

    candidate_id: str = ""
    full_name: str
    email: str
    phone: str | None
    skills: list[str]
    work_history: list[WorkExperience]
    education: list[EducationEntry]
    certifications: list[str]
    years_experience_total: float = 0.0

    def experience_summary(self) -> str:
        """Concatenate work history into a plain-text summary for embedding."""
        parts = []
        for job in self.work_history:
            bullets = " ".join(job.description)
            parts.append(
                f"{job.title} at {job.company} ({job.start_date} - {job.end_date or 'present'}): "
                f"{bullets}".strip()
            )
        return " ".join(parts)

    def education_summary(self) -> str:
        """Concatenate education history into a plain-text summary for embedding.

        Every EducationEntry field is optional, so this assembles only the
        pieces actually present rather than assuming a fully-populated entry.
        """
        parts = []
        for entry in self.education:
            pieces = []
            if entry.degree:
                pieces.append(entry.degree)
            if entry.field_of_study:
                pieces.append(f"in {entry.field_of_study}")
            if entry.institution:
                pieces.append(f"from {entry.institution}")
            if entry.start_date:
                pieces.append(f"({entry.start_date} - {entry.end_date or 'present'})")
            elif entry.end_date:
                pieces.append(f"({entry.end_date})")
            summary = " ".join(pieces)
            if entry.description:
                bullets = " ".join(entry.description)
                summary = f"{summary}: {bullets}" if summary else bullets
            if summary:
                parts.append(summary)
        return " ".join(parts)
