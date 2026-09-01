import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from personal_ai.application.experience.extractor import ExperienceExtractor
from personal_ai.domain.experience import (
    ClassificationResult,
    ExperienceExtractionResult,
    ExperienceImportance,
    ExperienceLifecycle,
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


# ==============================================================================
# PR #14 Specific Tests: Memory Importance, Lifecycle, and Qualifier Preservation
# ==============================================================================

@pytest.mark.asyncio
async def test_pr14_example_a_identity_fact_stable_high() -> None:
    """PR #14 Example A: 'My name is Shoaib.' -> FACT, STABLE, HIGH."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "Name is Shoaib",
        "type": "FACT",
        "domain": "personal",
        "importance": "HIGH",
        "lifecycle": "STABLE",
        "confidence": 0.98,
        "reasoning": "Core personal identity fact.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="gemini", model="gemini-3.6-flash", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.FACT, importance=1.0, confidence=0.99)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("My name is Shoaib.", classification=classification)

    assert res.success is True
    assert res.content == "Name is Shoaib"
    assert res.type == ExperienceType.FACT
    assert res.importance == ExperienceImportance.HIGH
    assert res.lifecycle == ExperienceLifecycle.STABLE
    assert res.confidence == 0.98


@pytest.mark.asyncio
async def test_pr14_example_b_goal_stable_high() -> None:
    """PR #14 Example B: 'I want to reach 30 LPA.' -> GOAL, STABLE, HIGH."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "Wants to reach 30 LPA salary",
        "type": "GOAL",
        "domain": "career",
        "importance": "HIGH",
        "lifecycle": "STABLE",
        "confidence": 0.95,
        "reasoning": "High-priority career aspiration.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="gemini", model="gemini-3.6-flash", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.9, confidence=0.95)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I want to reach 30 LPA.", classification=classification)

    assert res.success is True
    assert res.content == "Wants to reach 30 LPA salary"
    assert res.type == ExperienceType.GOAL
    assert res.importance == ExperienceImportance.HIGH
    assert res.lifecycle == ExperienceLifecycle.STABLE


@pytest.mark.asyncio
async def test_pr14_example_c_habit_recurring_preserves_qualifier() -> None:
    """PR #14 Example C: 'I usually go to the gym at 6 PM.' -> HABIT, RECURRING, MEDIUM/HIGH, preserves 'usually'."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "Usually goes to the gym around 6 PM",
        "type": "HABIT",
        "domain": "fitness",
        "importance": "MEDIUM",
        "lifecycle": "RECURRING",
        "confidence": 0.93,
        "reasoning": "Recurring workout habit with probabilistic qualifier.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="gemini", model="gemini-3.6-flash", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.HABIT, importance=0.7, confidence=0.92)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I usually go to the gym at 6 PM.", classification=classification)

    assert res.success is True
    assert "usually" in res.content.lower() or "around" in res.content.lower()
    assert "every day" not in res.content.lower()
    assert res.type == ExperienceType.HABIT
    assert res.importance in (ExperienceImportance.MEDIUM, ExperienceImportance.HIGH)
    assert res.lifecycle == ExperienceLifecycle.RECURRING


@pytest.mark.asyncio
async def test_pr14_example_d_state_temporary_low() -> None:
    """PR #14 Example D: 'I am tired today.' -> STATE, TEMPORARY, LOW."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "Feeling tired today",
        "type": "STATE",
        "domain": "personal",
        "importance": "LOW",
        "lifecycle": "TEMPORARY",
        "confidence": 0.90,
        "reasoning": "Fleeting daily physical/emotional condition.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="gemini", model="gemini-3.6-flash", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.STATE, importance=0.3, confidence=0.85)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I am tired today.", classification=classification)

    assert res.success is True
    assert res.content == "Feeling tired today"
    assert res.type == ExperienceType.STATE
    assert res.importance == ExperienceImportance.LOW
    assert res.lifecycle == ExperienceLifecycle.TEMPORARY


@pytest.mark.asyncio
async def test_pr14_example_e_preference_stable_medium() -> None:
    """PR #14 Example E: 'I like playing volleyball.' -> PREFERENCE, STABLE, MEDIUM."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "Likes playing volleyball",
        "type": "PREFERENCE",
        "domain": "hobbies",
        "importance": "MEDIUM",
        "lifecycle": "STABLE",
        "confidence": 0.94,
        "reasoning": "Sport hobby preference.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="gemini", model="gemini-3.6-flash", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.PREFERENCE, importance=0.6, confidence=0.90)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I like playing volleyball.", classification=classification)

    assert res.success is True
    assert res.content == "Likes playing volleyball"
    assert res.type == ExperienceType.PREFERENCE
    assert res.importance == ExperienceImportance.MEDIUM
    assert res.lifecycle == ExperienceLifecycle.STABLE


@pytest.mark.asyncio
async def test_pr14_example_f_event_time_bound_high() -> None:
    """PR #14 Example F: 'I have an interview tomorrow.' -> EVENT, TIME_BOUND, HIGH."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "content": "Has an interview tomorrow",
        "type": "EVENT",
        "domain": "career",
        "importance": "HIGH",
        "lifecycle": "TIME_BOUND",
        "confidence": 0.96,
        "reasoning": "Upcoming scheduled interview event.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="gemini", model="gemini-3.6-flash", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.EVENT, importance=0.85, confidence=0.95)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I have an interview tomorrow.", classification=classification)

    assert res.success is True
    assert res.content == "Has an interview tomorrow"
    assert res.type == ExperienceType.EVENT
    assert res.importance == ExperienceImportance.HIGH
    assert res.lifecycle == ExperienceLifecycle.TIME_BOUND


@pytest.mark.asyncio
async def test_pr14_critical_negative_qualifier_preservation() -> None:
    """PR #14 Requirement 11 (Critical Negative Test):
    Asserts system does NOT transform approximate/probabilistic statement into absolute fact.
    """
    mock_llm = MagicMock(spec=LLMClient)
    # Extractor faithfully preserves qualifier
    payload = {
        "content": "Usually goes to the gym around 6 PM",
        "type": "HABIT",
        "domain": "fitness",
        "importance": "MEDIUM",
        "lifecycle": "RECURRING",
        "confidence": 0.95,
        "reasoning": "Preserved 'usually' and approximate timing.",
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content=json.dumps(payload), provider="gemini", model="gemini-3.6-flash", latency_ms=10.0)
    )
    classification = ClassificationResult(is_experience=True, type=ExperienceType.HABIT, importance=0.7, confidence=0.90)

    extractor = ExperienceExtractor(llm_client=mock_llm)
    res = await extractor.extract("I usually go to the gym at 6 PM.", classification=classification)

    assert res.success is True
    # MUST NOT be absolute every day statement
    assert "every day" not in res.content.lower()
    assert "daily" not in res.content.lower()
    assert "always" not in res.content.lower()
    assert "usually" in res.content.lower() or "around" in res.content.lower()

