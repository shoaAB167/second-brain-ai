import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from personal_ai.application.experience.classifier import ExperienceClassifier
from personal_ai.domain.experience import ClassificationResult, ExperienceType
from personal_ai.llm.client import LLMClient
from personal_ai.llm.exceptions import LLMRateLimitException
from personal_ai.llm.models import LLMResponse


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "category_type",
    [
        ExperienceType.GOAL,
        ExperienceType.DECISION,
        ExperienceType.PREFERENCE,
        ExperienceType.FACT,
        ExperienceType.EVENT,
        ExperienceType.RELATIONSHIP,
        ExperienceType.EMOTION_STATE,
        ExperienceType.HABIT,
        ExperienceType.PROJECT,
        ExperienceType.OTHER,
    ],
)
async def test_experience_classifier_all_taxonomy_types(category_type: ExperienceType) -> None:
    """Verify ExperienceClassifier correctly parses all taxonomy categories."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": True,
        "type": category_type.value,
        "importance": 0.85,
        "confidence": 0.95,
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(payload),
            provider="openai",
            model="gpt-4o-mini",
            latency_ms=10.0,
        )
    )

    classifier = ExperienceClassifier(llm_client=mock_llm)
    result = await classifier.classify("Sample text")

    assert isinstance(result, ClassificationResult)
    assert result.is_experience is True
    assert result.type == category_type
    assert result.importance == 0.85
    assert result.confidence == 0.95
    assert result.raw_model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_experience_classifier_invalid_type_fails_closed() -> None:
    """25 & B3. Verify invalid experience type fails closed (returns fallback is_experience=False)."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": True,
        "type": "INVALID_UNKNOWN_TYPE_123",
        "importance": 0.9,
        "confidence": 0.9,
    }
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(payload),
            provider="openai",
            model="gpt-4o-mini",
            latency_ms=10.0,
        )
    )

    classifier = ExperienceClassifier(llm_client=mock_llm)
    result = await classifier.classify("Some message")

    assert result.is_experience is False
    assert result.type is None
    assert result.raw_model == "fallback_parse_error"


def test_classification_result_type_null_when_is_experience_false() -> None:
    """26. Verify is_experience=False requires type=None."""
    # Valid: is_experience=False, type=None
    res = ClassificationResult(is_experience=False, type=None, importance=0.1, confidence=0.9)
    assert res.is_experience is False
    assert res.type is None

    # Invalid: is_experience=False, type=GOAL -> raises ValueError
    with pytest.raises(ValueError, match="is_experience is False"):
        ClassificationResult(is_experience=False, type=ExperienceType.GOAL, importance=0.1, confidence=0.9)


def test_classification_result_score_range_validation() -> None:
    """27 & 28. Verify invalid importance or confidence scores are rejected."""
    # Valid
    res = ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.0, confidence=1.0)
    assert res.importance == 0.0
    assert res.confidence == 1.0

    # Importance out of range (> 1.0)
    with pytest.raises(ValueError, match="less than or equal to 1"):
        ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=1.5, confidence=0.8)

    # Confidence out of range (< 0.0)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.5, confidence=-0.1)


@pytest.mark.asyncio
async def test_experience_classifier_handles_malformed_json_gracefully() -> None:
    """Verify ExperienceClassifier returns safe fallback on invalid JSON."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(
            content="This is not valid JSON string",
            provider="openai",
            model="gpt-4o-mini",
            latency_ms=10.0,
        )
    )

    classifier = ExperienceClassifier(llm_client=mock_llm)
    result = await classifier.classify("Some message")

    assert result.is_experience is False
    assert result.type is None
    assert result.importance == 0.0
    assert result.confidence == 0.0
    assert result.raw_model == "fallback_parse_error"


@pytest.mark.asyncio
async def test_experience_classifier_handles_llm_exception_gracefully() -> None:
    """29. Verify ExperienceClassifier returns safe fallback on LLMException without raising."""
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        side_effect=LLMRateLimitException("Rate limit exceeded")
    )

    classifier = ExperienceClassifier(llm_client=mock_llm)
    result = await classifier.classify("Some message")

    assert result.is_experience is False
    assert result.type is None
    assert result.importance == 0.0
    assert result.confidence == 0.0
    assert result.raw_model == "fallback_llm_error"
