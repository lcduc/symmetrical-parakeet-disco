import numpy as np
import pytest

from models.candidate import CandidateProfile, EducationEntry, WorkExperience
from models.job import JobRequirements
from services.matching_service import MatchingService


class _FakeEmbeddings:
    """Stand-in for EmbeddingService: returns a fixed vector per cache key, keyed by exact text."""

    def __init__(self, vectors_by_text: dict):
        self._vectors_by_text = vectors_by_text

    async def embed_cached(self, _key: str, text: str) -> np.ndarray:
        return self._vectors_by_text[text]


class _FakeLLM:
    """Stand-in for LLMAdapter: returns a fixed explanation without any I/O."""

    async def complete(self, _prompt: str, max_tokens: int = 1024, temperature: float | None = None) -> str:
        return "  This candidate is a strong fit.  "


class _FailingLLM:
    """Stand-in for LLMAdapter: simulates a transient provider error on every call."""

    async def complete(self, _prompt: str, max_tokens: int = 1024, temperature: float | None = None) -> str:
        raise RuntimeError("provider timeout")


def _make_candidate(**overrides) -> CandidateProfile:
    defaults = {
        "candidate_id": "cand-1",
        "full_name": "Jane Doe",
        "email": "jane@example.com",
        "phone": None,
        "skills": ["Python", "SQL"],
        "work_history": [
            WorkExperience(
                company="Acme",
                title="Engineer",
                start_date="2020-01",
                end_date=None,
                description=["Built backend services in Python."],
            )
        ],
        "education": [EducationEntry(level="bachelor")],
        "certifications": ["PMP"],
        "years_experience_total": 5.0,
    }
    defaults.update(overrides)
    return CandidateProfile(**defaults)


def _make_job(**overrides) -> JobRequirements:
    defaults = {
        "job_id": "job-1",
        "title": "Software Engineer",
        "jd_text": "Build things.",
        "min_years_experience": 3.0,
        "required_education_level": "bachelor",
        "responsibilities_text": "Build backend services in Python.",
    }
    defaults.update(overrides)
    return JobRequirements(**defaults)


def _service() -> MatchingService:
    return MatchingService(embeddings=None, llm=None)


def test_insufficient_years_experience_fails_below_tolerance():
    candidate = _make_candidate(years_experience_total=1.0)
    job = _make_job(min_years_experience=3.0)

    passed, notes, experience_flag = _service().apply_hard_filters(candidate, job)

    assert passed is False
    assert "3.0" in notes
    assert experience_flag == ""


def test_years_experience_within_tolerance_passes_but_is_flagged():
    candidate = _make_candidate(years_experience_total=1.0)
    job = _make_job(min_years_experience=2.0)

    passed, _notes, experience_flag = _service().apply_hard_filters(candidate, job)

    assert passed is True
    assert experience_flag != ""
    assert "1.0" in experience_flag
    assert "2.0" in experience_flag


def test_insufficient_education_level_fails():
    candidate = _make_candidate(education=[EducationEntry(level="associate")])
    job = _make_job(required_education_level="bachelor")

    passed, notes, experience_flag = _service().apply_hard_filters(candidate, job)

    assert passed is False
    assert "bachelor" in notes
    assert experience_flag == ""


def test_all_requirements_met_passes_without_flag():
    candidate = _make_candidate()
    job = _make_job()

    passed, notes, experience_flag = _service().apply_hard_filters(candidate, job)

    assert passed is True
    assert notes == "Meets all hard requirements."
    assert experience_flag == ""


@pytest.mark.asyncio
async def test_score_soft_fit_averages_best_match_per_job_requirement():
    candidate = _make_candidate(
        work_history=[
            WorkExperience(
                company="Acme",
                title="Engineer",
                start_date="2020-01",
                end_date=None,
                description=["bullet A", "bullet B"],
            )
        ]
    )
    job = _make_job(responsibilities_text="requirement one\nrequirement two")

    vectors = {
        "bullet A": np.array([1.0, 0.0], dtype=np.float32),
        "bullet B": np.array([0.0, 1.0], dtype=np.float32),
        "requirement one": np.array([1.0, 0.0], dtype=np.float32),  # best match: bullet A, cos = 1.0
        "requirement two": np.array([0.0, 1.0], dtype=np.float32),  # best match: bullet B, cos = 1.0
    }
    service = MatchingService(embeddings=_FakeEmbeddings(vectors), llm=None)

    score = await service.score_soft_fit(candidate, job)

    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_score_soft_fit_is_zero_with_no_experience_bullets():
    candidate = _make_candidate(work_history=[], years_experience_total=0.0)
    job = _make_job(responsibilities_text="requirement one")

    service = MatchingService(embeddings=_FakeEmbeddings({}), llm=None)

    score = await service.score_soft_fit(candidate, job)

    assert score == 0.0


@pytest.mark.asyncio
async def test_match_caps_overall_score_when_hard_filters_fail():
    candidate = _make_candidate(years_experience_total=0.5)  # below tolerance floor for min_years=3.0
    job = _make_job()

    same_vector = np.array([1.0, 0.0], dtype=np.float32)
    vectors = {
        "Built backend services in Python.": same_vector,
        "Build backend services in Python.": same_vector,
    }
    service = MatchingService(embeddings=_FakeEmbeddings(vectors), llm=_FakeLLM(), hard_filter_soft_cap=0.4)

    result = await service.match(candidate, job)

    assert result.passed_hard_filters is False
    assert result.soft_fit_score == pytest.approx(1.0)
    assert result.overall_score == pytest.approx(0.4)
    assert result.explanation == "This candidate is a strong fit."


@pytest.mark.asyncio
async def test_match_leaves_overall_score_uncapped_when_hard_filters_pass():
    candidate = _make_candidate()
    job = _make_job()

    same_vector = np.array([1.0, 0.0], dtype=np.float32)
    vectors = {
        "Built backend services in Python.": same_vector,
        "Build backend services in Python.": same_vector,
    }
    service = MatchingService(embeddings=_FakeEmbeddings(vectors), llm=_FakeLLM(), hard_filter_soft_cap=0.4)

    result = await service.match(candidate, job)

    assert result.passed_hard_filters is True
    assert result.overall_score == pytest.approx(1.0)
    assert result.experience_flag == ""
    assert result.explanation == "This candidate is a strong fit."


@pytest.mark.asyncio
async def test_match_falls_back_to_deterministic_explanation_on_llm_failure():
    candidate = _make_candidate()
    job = _make_job()

    same_vector = np.array([1.0, 0.0], dtype=np.float32)
    vectors = {
        "Built backend services in Python.": same_vector,
        "Build backend services in Python.": same_vector,
    }
    service = MatchingService(embeddings=_FakeEmbeddings(vectors), llm=_FailingLLM(), hard_filter_soft_cap=0.4)

    result = await service.match(candidate, job)

    assert result.passed_hard_filters is True
    assert result.overall_score == pytest.approx(1.0)
    assert "Meets all hard requirements." in result.explanation
