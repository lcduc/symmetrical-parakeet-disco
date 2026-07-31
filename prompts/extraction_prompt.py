import json

from pydantic import ValidationError

from models.candidate import CandidateProfile

SCHEMA_JSON = CandidateProfile.model_json_schema()

FORBIDDEN_FIELDS_NOTICE = (
    "Do NOT extract or infer date of birth, age, photo, nationality, "
    "marital status, or gender, even if present in the source text. "
    "These fields must be omitted entirely from your response."
)

def build_extraction_prompt(resume_text: str) -> str:
    """Build the initial structured-extraction prompt for a resume's raw text."""
    return (
        "Extract a candidate profile from the resume text below, strictly "
        "conforming to this JSON schema. Respond with JSON only, no "
        "commentary or markdown fences.\n\n"
        f"Schema:\n{json.dumps(SCHEMA_JSON)}\n\n"
        f"{FORBIDDEN_FIELDS_NOTICE}\n\n"
        f"Resume text:\n{resume_text}"
    )


def build_correction_prompt(resume_text: str, validation_error: ValidationError) -> str:
    """Build a correction prompt listing the specific validation errors to fix."""
    field_errors = "\n".join(
        f"- {'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        for error in validation_error.errors()
    )
    return (
        "Your previous response did not match the required schema. Fix the "
        "following field errors and respond with corrected JSON only, no "
        "commentary or markdown fences.\n\n"
        f"Field errors:\n{field_errors}\n\n"
        f"Schema:\n{json.dumps(SCHEMA_JSON)}\n\n"
        f"{FORBIDDEN_FIELDS_NOTICE}\n\n"
        f"Resume text:\n{resume_text}"
    )
