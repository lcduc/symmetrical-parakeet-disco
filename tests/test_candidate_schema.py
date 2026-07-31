from models.candidate import CandidateProfile, EducationEntry, WorkExperience

EXCLUDED_FIELDS = {
    "date_of_birth",
    "age",
    "photo",
    "nationality",
    "marital_status",
    "gender",
}


def test_candidate_profile_excludes_pii_fields():
    present = EXCLUDED_FIELDS & set(CandidateProfile.model_fields)
    assert not present, f"CandidateProfile must not define PII fields: {present}"


def _sample_profile() -> CandidateProfile:
    return CandidateProfile(
        full_name="Jane Doe",
        email="jane@example.com",
        phone=None,
        skills=["python", "sql"],
        work_history=[
            WorkExperience(
                company="Acme Corp",
                title="Software Engineer",
                start_date="2020-01",
                end_date="2023-06",
                description=["Built backend services.", "Led migration to microservices."],
            )
        ],
        education=[
            EducationEntry(
                institution="State University",
                degree="B.Sc. Computer Science",
                field_of_study="Computer Science",
                level="Bachelor",
                start_date="Sep 2015",
                end_date="Jun 2019",
            )
        ],
        certifications=["AWS Certified Developer"],
        years_experience_total=3.5,
    )


def test_experience_summary_returns_non_empty_string():
    profile = _sample_profile()
    assert profile.experience_summary() != ""


def test_education_summary_returns_non_empty_string():
    profile = _sample_profile()
    assert profile.education_summary() != ""


def test_education_entry_accepts_end_date_only():
    entry = EducationEntry(end_date="2019")
    assert entry.institution is None
    assert entry.degree is None
    assert entry.start_date is None
    assert entry.description == []


def test_education_summary_handles_end_date_only():
    profile = _sample_profile()
    profile.education = [EducationEntry(end_date="2019")]
    assert profile.education_summary() == "(2019)"


def test_education_summary_handles_date_range():
    profile = _sample_profile()
    profile.education = [EducationEntry(start_date="Sep 2022", end_date="Aug 2025")]
    assert profile.education_summary() == "(Sep 2022 - Aug 2025)"


def test_work_experience_keeps_present_as_literal_text():
    job = WorkExperience(company="Acme", title="Engineer", start_date="Oct 2024", end_date="present")
    assert job.end_date == "present"


def test_education_keeps_ongoing_phrasing_as_literal_text():
    for phrasing in ["Present", "current", "CURRENTLY", "ongoing", "Now"]:
        entry = EducationEntry(start_date="2022", end_date=phrasing)
        assert entry.end_date == phrasing


def test_experience_summary_keeps_present_intact():
    profile = _sample_profile()
    profile.work_history = [
        WorkExperience(company="Acme", title="Engineer", start_date="Oct 2024", end_date="present", description=[])
    ]
    assert "Oct 2024 - present" in profile.experience_summary()


def test_education_summary_keeps_present_intact():
    profile = _sample_profile()
    profile.education = [EducationEntry(start_date="Sep 2022", end_date="present")]
    assert profile.education_summary() == "(Sep 2022 - present)"


def test_experience_summary_falls_back_to_present_when_end_date_truly_absent():
    profile = _sample_profile()
    profile.work_history = [
        WorkExperience(company="Acme", title="Engineer", start_date="2024", end_date=None, description=[])
    ]
    assert "2024 - present" in profile.experience_summary()
