import uuid
from unittest.mock import AsyncMock, MagicMock
import pytest

from personal_ai.application.experience import (
    AIExperiencePromotionStrategy,
    ExperienceClassifier,
)
from personal_ai.db.models import Message, MessageRole
from personal_ai.domain.experience import ClassificationResult, ExperienceType


@pytest.mark.asyncio
async def test_ai_promotion_policy_is_experience_false_rejected() -> None:
    """Verify is_experience=False evaluates to should_promote=False."""
    mock_classifier = MagicMock(spec=ExperienceClassifier)
    mock_classifier.classify = AsyncMock(
        return_value=ClassificationResult(
            is_experience=False,
            type=None,
            importance=0.1,
            confidence=0.9,
        )
    )

    strategy = AIExperiencePromotionStrategy(
        classifier=mock_classifier,
        min_confidence=0.70,
        min_importance=0.50,
    )
    user_msg = Message(id=uuid.uuid4(), role=MessageRole.USER, content="What is Java?")

    should_promote, res = await strategy.evaluate_async(user_msg)
    assert should_promote is False
    assert res.is_experience is False


@pytest.mark.asyncio
async def test_ai_promotion_policy_low_confidence_rejected() -> None:
    """Verify is_experience=True with confidence < min_confidence evaluates to should_promote=False."""
    mock_classifier = MagicMock(spec=ExperienceClassifier)
    mock_classifier.classify = AsyncMock(
        return_value=ClassificationResult(
            is_experience=True,
            type=ExperienceType.GOAL,
            importance=0.80,
            confidence=0.65,  # < 0.70 threshold
        )
    )

    strategy = AIExperiencePromotionStrategy(
        classifier=mock_classifier,
        min_confidence=0.70,
        min_importance=0.50,
    )
    user_msg = Message(id=uuid.uuid4(), role=MessageRole.USER, content="I might learn Go.")

    should_promote, res = await strategy.evaluate_async(user_msg)
    assert should_promote is False


@pytest.mark.asyncio
async def test_ai_promotion_policy_low_importance_rejected() -> None:
    """Verify is_experience=True with importance < min_importance evaluates to should_promote=False."""
    mock_classifier = MagicMock(spec=ExperienceClassifier)
    mock_classifier.classify = AsyncMock(
        return_value=ClassificationResult(
            is_experience=True,
            type=ExperienceType.OTHER,
            importance=0.40,  # < 0.50 threshold
            confidence=0.90,
        )
    )

    strategy = AIExperiencePromotionStrategy(
        classifier=mock_classifier,
        min_confidence=0.70,
        min_importance=0.50,
    )
    user_msg = Message(id=uuid.uuid4(), role=MessageRole.USER, content="I drank tea today.")

    should_promote, res = await strategy.evaluate_async(user_msg)
    assert should_promote is False


@pytest.mark.asyncio
async def test_ai_promotion_policy_pass_both_thresholds_promoted() -> None:
    """Verify is_experience=True meeting both thresholds evaluates to should_promote=True."""
    mock_classifier = MagicMock(spec=ExperienceClassifier)
    mock_classifier.classify = AsyncMock(
        return_value=ClassificationResult(
            is_experience=True,
            type=ExperienceType.GOAL,
            importance=0.90,  # >= 0.50
            confidence=0.95,  # >= 0.70
        )
    )

    strategy = AIExperiencePromotionStrategy(
        classifier=mock_classifier,
        min_confidence=0.70,
        min_importance=0.50,
    )
    user_msg = Message(id=uuid.uuid4(), role=MessageRole.USER, content="I've decided to focus on AI engineering.")

    should_promote, res = await strategy.evaluate_async(user_msg)
    assert should_promote is True
    assert res.type == ExperienceType.GOAL
