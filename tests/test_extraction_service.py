from unittest.mock import AsyncMock, patch

import pytest

from services.extraction_service import ExtractionService

VALID_PROFILE_JSON = """
{
    "full_name": "Jane Doe",
    "email": "jane@example.com",
    "phone": null,
    "skills": ["python", "sql"],
    "work_history": [],
    "education": [],
    "certifications": [],
    "years_experience_total": 3.5
}
"""

INVALID_PROFILE_JSON = '{"full_name": "Jane Doe"}'

PROFILE_JSON_WITH_UNLABELED_DEGREE = """
{
    "full_name": "Jane Doe",
    "email": "jane@example.com",
    "phone": null,
    "skills": ["python"],
    "work_history": [],
    "education": [{"degree": "BSc", "level": null, "institution": null, "field_of_study": null,
                   "start_date": null, "end_date": null, "description": []}],
    "certifications": [],
    "years_experience_total": 3.5
}
"""


@pytest.mark.asyncio
async def test_extract_succeeds_on_first_attempt():
    llm = AsyncMock()
    llm.complete_structured.return_value = VALID_PROFILE_JSON

    profile, attempts = await ExtractionService(llm).extract("resume text")

    assert profile is not None
    assert profile.full_name == "Jane Doe"
    assert attempts == 1
    llm.complete_structured.assert_awaited_once()


@pytest.mark.asyncio
async def test_extract_fails_once_then_succeeds_on_retry():
    llm = AsyncMock()
    llm.complete_structured.side_effect = [INVALID_PROFILE_JSON, VALID_PROFILE_JSON]

    profile, attempts = await ExtractionService(llm).extract("resume text")

    assert profile is not None
    assert attempts == 2
    assert llm.complete_structured.await_count == 2

    first_prompt = llm.complete_structured.await_args_list[0].args[0]
    second_prompt = llm.complete_structured.await_args_list[1].args[0]
    assert first_prompt != second_prompt
    assert "field errors" in second_prompt.lower()


@pytest.mark.asyncio
async def test_extract_exhausts_retries_and_returns_none():
    llm = AsyncMock()
    llm.complete_structured.return_value = INVALID_PROFILE_JSON

    profile, attempts = await ExtractionService(llm, max_retries=3).extract("resume text")

    assert profile is None
    assert attempts == 3
    assert llm.complete_structured.await_count == 3


@pytest.mark.asyncio
async def test_extract_retries_after_transient_llm_error():
    llm = AsyncMock()
    llm.complete_structured.side_effect = [RuntimeError("rate limited"), VALID_PROFILE_JSON]

    with patch("services.extraction_service.asyncio.sleep", new_callable=AsyncMock):
        profile, attempts = await ExtractionService(llm, max_retries=3).extract("resume text")

    assert profile is not None
    assert attempts == 2
    assert llm.complete_structured.await_count == 2


@pytest.mark.asyncio
async def test_extract_reads_max_retries_from_settings_by_default():
    llm = AsyncMock()
    llm.complete_structured.return_value = INVALID_PROFILE_JSON

    profile, attempts = await ExtractionService(llm).extract("resume text")

    assert profile is None
    from config.settings import settings

    assert attempts == settings.max_extraction_retries


@pytest.mark.asyncio
async def test_extract_fills_missing_education_level_from_degree():
    llm = AsyncMock()
    llm.complete_structured.return_value = PROFILE_JSON_WITH_UNLABELED_DEGREE

    profile, _ = await ExtractionService(llm).extract("resume text")

    assert profile is not None
    assert profile.education[0].level == "bachelor"
