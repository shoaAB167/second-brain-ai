import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from personal_ai.application.experience.extractor import ExperienceExtractor
from personal_ai.domain.experience import (
    ClassificationResult,
    ExperienceExtractionResult,
    ExperienceType,
)
from personal_ai.llm.client import LLMClient
from personal_ai.llm.exceptions import LLMRateLimitException
from personal_ai.llm.models import LLMResponse


@pytest.mark.asyncio
async def test_case_a_successful_fact_extraction() -> None:
    """Requirement 6A: Successful FACT extraction -> returns success=True and structured content."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "Works primarily with Java and Spring Boot",
        "domain": "career/technology",
        "status": "active",
        "confidence": 0.95,
        "reasoning": "Extracted tech stack facts.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.FACT, importance=0.8, confidence=0.95)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I work primarily with Java and Spring Boot.", classification=classification)

    assert isinstance(res, ExperienceExtractionResult)
    assert res.success is True
    assert res.content == "Works primarily with Java and Spring Boot"
    assert res.domain == "career/technology"
    assert res.confidence == 0.95


@pytest.mark.asyncio
async def test_case_a_successful_preference_extraction() -> None:
    """Requirement 6A: Successful PREFERENCE extraction."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "Prefers remote work",
        "domain": "work",
        "status": "active",
        "confidence": 0.92,
        "reasoning": "User preference extracted.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.PREFERENCE, importance=0.75, confidence=0.90)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I prefer remote work.", classification=classification)

    assert res.success is True
    assert res.content == "Prefers remote work"
    assert res.domain == "work"


@pytest.mark.asyncio
async def test_case_a_successful_goal_extraction() -> None:
    """Requirement 6A: Successful GOAL extraction."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "Reach 30 LPA salary target",
        "domain": "career",
        "status": "active",
        "confidence": 0.96,
        "reasoning": "Salary goal extracted.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.90, confidence=0.95)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I want to reach 30 LPA.", classification=classification)

    assert res.success is True
    assert res.content == "Reach 30 LPA salary target"
    assert res.status == "active"


@pytest.mark.asyncio
async def test_case_a_successful_habit_extraction() -> None:
    """Requirement 6A: Successful HABIT extraction."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "Works out five days a week",
        "domain": "fitness",
        "status": "active",
        "confidence": 0.94,
        "reasoning": "Fitness habit extracted.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.HABIT, importance=0.70, confidence=0.90)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I work out five days a week.", classification=classification)

    assert res.success is True
    assert res.content == "Works out five days a week"
    assert res.domain == "fitness"


@pytest.mark.asyncio
async def test_case_a_successful_project_extraction() -> None:
    """Requirement 6A: Successful PROJECT extraction."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "Building a personal AI assistant",
        "domain": "projects",
        "status": "active",
        "confidence": 0.93,
        "reasoning": "Software project extracted.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.PROJECT, importance=0.85, confidence=0.92)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I'm building a personal AI assistant.", classification=classification)

    assert res.success is True
    assert res.content == "Building a personal AI assistant"


@pytest.mark.asyncio
async def test_no_hallucination() -> None:
    """Requirement 6: Extractor does NOT invent unsupported attributes."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "Wants a better job",
        "domain": "career",
        "status": "active",
        "confidence": 0.90,
        "reasoning": "Extracted unspecific job aspiration without unsupported details.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.8, confidence=0.9)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I want a better job.", classification=classification)

    assert res.success is True
    assert res.content == "Wants a better job"
    assert "30 LPA" not in res.content
    assert "FAANG" not in res.content


@pytest.mark.asyncio
async def test_case_b_extraction_llm_failure_returns_success_false() -> None:
    """Requirement 6B: LLM exception during extraction returns success=False."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(side_effect=LLMRateLimitException("Rate limit exceeded"))
    classification = ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.8, confidence=0.9)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I want to learn AI.", classification=classification)

    assert res.success is False
    assert res.content is None
    assert res.confidence == 0.0
    assert res.raw_model == "fallback_llm_error"


@pytest.mark.asyncio
async def test_case_c_invalid_json_returns_success_false() -> None:
    """Requirement 6C: Invalid extraction JSON returns success=False."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content="Malformed json {", provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.8, confidence=0.9)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I want to learn AI.", classification=classification)

    assert res.success is False
    assert res.content is None
    assert res.confidence == 0.0
    assert res.raw_model == "fallback_parse_error"


@pytest.mark.asyncio
async def test_case_d_invalid_extraction_schema_returns_success_false() -> None:
    """Requirement 6D: Invalid extraction schema (e.g. invalid string confidence) returns success=False."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "Learns AI",
        "confidence": "high",  # invalid string score
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.8, confidence=0.9)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I want to learn AI.", classification=classification)

    assert res.success is False
    assert res.content is None
    assert res.confidence == 0.0
    assert res.raw_model == "fallback_parse_error"


@pytest.mark.asyncio
async def test_case_e_empty_extracted_content_returns_success_false() -> None:
    """Requirement 6E: Empty extracted content returns success=False."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "  ",  # Whitespace-only content
        "confidence": 0.9,
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.8, confidence=0.9)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I want to learn AI.", classification=classification)

    assert res.success is False
    assert res.content is None
    assert res.confidence == 0.0
    assert res.raw_model == "fallback_parse_error"


@pytest.mark.asyncio
async def test_classifier_false_skips_extractor_call() -> None:
    """Verify classifier evaluating is_experience=False returns success=False immediately without calling LLM."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock()

    classification = ClassificationResult(is_experience=False, type=None, importance=0.1, confidence=0.99)
    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("What is dependency injection?", classification=classification)

    assert res.success is False
    assert res.content is None
    assert res.confidence == 0.0
    mock_llm.generate_response.assert_not_called()
