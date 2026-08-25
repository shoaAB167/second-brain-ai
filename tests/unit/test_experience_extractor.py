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
from personal_ai.llm.models import LLMMessage, LLMResponse


@pytest.mark.asyncio
async def test_case_1_fact_extraction() -> None:
    """Requirement 17.1: FACT extraction -> content represents user's technology experience."""
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
    assert res.content == "Works primarily with Java and Spring Boot"
    assert res.type == ExperienceType.FACT
    assert res.domain == "career/technology"
    assert res.confidence == 0.95


@pytest.mark.asyncio
async def test_case_2_preference_extraction() -> None:
    """Requirement 17.2: PREFERENCE extraction -> preference represented accurately."""
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

    assert res.content == "Prefers remote work"
    assert res.type == ExperienceType.PREFERENCE
    assert res.domain == "work"


@pytest.mark.asyncio
async def test_case_3_goal_extraction() -> None:
    """Requirement 17.3: GOAL extraction -> goal represented accurately."""
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

    assert res.content == "Reach 30 LPA salary target"
    assert res.type == ExperienceType.GOAL
    assert res.status == "active"


@pytest.mark.asyncio
async def test_case_4_habit_extraction() -> None:
    """Requirement 17.4: HABIT extraction -> habit represented accurately."""
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

    assert res.content == "Works out five days a week"
    assert res.type == ExperienceType.HABIT
    assert res.domain == "fitness"


@pytest.mark.asyncio
async def test_case_5_project_extraction() -> None:
    """Requirement 17.5: PROJECT extraction -> project represented accurately."""
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

    assert res.content == "Building a personal AI assistant"
    assert res.type == ExperienceType.PROJECT


@pytest.mark.asyncio
async def test_case_6_no_hallucination() -> None:
    """Requirement 17.6: No hallucination -> Extractor does NOT invent unsupported attributes."""
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

    assert res.content == "Wants a better job"
    assert "30 LPA" not in res.content
    assert "FAANG" not in res.content
    assert "Bangalore" not in res.content


@pytest.mark.asyncio
async def test_case_7_hypothetical() -> None:
    """Requirement 17.7: Hypothetical statement -> Extracted faithfully without fabricating active facts."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "Would choose Kafka for banking system architecture",
        "domain": "architecture",
        "status": "hypothetical",
        "confidence": 0.85,
        "reasoning": "Extracted architecture opinion.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.DECISION, importance=0.5, confidence=0.7)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("If I were building a banking system, I would use Kafka.", classification=classification)

    assert res.content == "Would choose Kafka for banking system architecture"


@pytest.mark.asyncio
async def test_case_8_invalid_json_fails_safely() -> None:
    """Requirement 17.8: Invalid JSON -> Extractor returns safe fallback."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content="Not a JSON string {", provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.8, confidence=0.9)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I want to learn AI.", classification=classification)

    assert res.content == ""
    assert res.confidence == 0.0
    assert res.raw_model == "fallback_parse_error"


@pytest.mark.asyncio
async def test_case_9_invalid_confidence_fails_safely() -> None:
    """Requirement 17.9: Invalid confidence score -> Extractor returns safe fallback."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "Learns AI",
        "type": "GOAL",
        "confidence": "high",  # invalid string score
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.8, confidence=0.9)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I want to learn AI.", classification=classification)

    assert res.content == ""
    assert res.confidence == 0.0
    assert res.raw_model == "fallback_parse_error"


@pytest.mark.asyncio
async def test_case_10_missing_content_fails_safely() -> None:
    """Requirement 17.10: Missing content -> Extractor returns safe fallback."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "",  # Empty content
        "type": "GOAL",
        "confidence": 0.9,
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="openai", model="gpt-4o-mini", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.8, confidence=0.9)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I want to learn AI.", classification=classification)

    assert res.content == ""
    assert res.confidence == 0.0
    assert res.raw_model == "fallback_parse_error"


@pytest.mark.asyncio
async def test_case_11_classifier_false_skips_extractor() -> None:
    """Requirement 17.11: Classifier says false -> Extractor returns fallback immediately without calling LLM."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock()

    classification = ClassificationResult(is_experience=False, type=None, importance=0.1, confidence=0.99)
    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("What is dependency injection?", classification=classification)

    assert res.content == ""
    assert res.confidence == 0.0
    mock_llm.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_case_12_llm_exception_fails_safely() -> None:
    """Requirement 17.12: LLM Exception -> Extractor returns safe fallback without crashing application."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(side_effect=LLMRateLimitException("Rate limit exceeded"))
    classification = ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.8, confidence=0.9)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I want to learn AI.", classification=classification)

    assert res.content == ""
    assert res.confidence == 0.0
    assert res.raw_model == "fallback_llm_error"
