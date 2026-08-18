import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from personal_ai.application.experience.classifier import ExperienceClassifier
from personal_ai.domain.experience import ClassificationResult, ExperienceType
from personal_ai.llm.client import LLMClient
from personal_ai.llm.exceptions import LLMException, LLMRateLimitException
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
    """Verify ExperienceClassifier correctly parses all 10 taxonomy categories."""
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
async def test_experience_classifier_informational_question_returns_false() -> None:
    """Verify informational question returns is_experience=False."""
    mock_llm = MagicMock(spec=LLMClient)
    payload = {
        "is_experience": False,
        "type": None,
        "importance": 0.05,
        "confidence": 0.99,
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
    result = await classifier.classify("What is cosine similarity?")

    assert result.is_experience is False
    assert result.type is None
    assert result.importance == 0.05
    assert result.confidence == 0.99


def test_classification_result_score_range_validation() -> None:
    """Verify ClassificationResult enforces 0.0 to 1.0 score boundaries."""
    # Valid
    res = ClassificationResult(is_experience=True, importance=0.0, confidence=1.0)
    assert res.importance == 0.0
    assert res.confidence == 1.0

    # Importance out of range (> 1.0)
    with pytest.raises(ValueError, match="less than or equal to 1"):
        ClassificationResult(is_experience=True, importance=1.5, confidence=0.8)

    # Confidence out of range (< 0.0)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        ClassificationResult(is_experience=True, importance=0.5, confidence=-0.1)


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
    """Verify ExperienceClassifier returns safe fallback on LLMException without raising."""
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
