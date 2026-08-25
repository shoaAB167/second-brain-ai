import uuid
from unittest.mock import AsyncMock, MagicMock
import pytest

from personal_ai.application.experience import (
    AIExperiencePromotionStrategy,
    DeterministicPromotionStrategy,
    ExperienceExtractor,
    ExperiencePromotionService,
    PromotionResult,
    RecordExperience,
)
from personal_ai.db.models import Message, MessageRole
from personal_ai.domain.experience import (
    ClassificationResult,
    Experience,
    ExperienceExtractionResult,
    ExperienceRepository,
    ExperienceSource,
    ExperienceStatus,
    ExperienceType,
)


def test_deterministic_strategy_only_promotes_user_messages_with_signal() -> None:
    """Verify DeterministicPromotionStrategy evaluates True ONLY for user messages with explicit signal."""
    strategy = DeterministicPromotionStrategy()

    user_msg = Message(id=uuid.uuid4(), role=MessageRole.USER, content="I decided to focus on AI.")
    assistant_msg = Message(id=uuid.uuid4(), role=MessageRole.ASSISTANT, content="AI is a great choice.")
    system_msg = Message(id=uuid.uuid4(), role=MessageRole.SYSTEM, content="System prompt.")

    # User message with signal = True -> True
    assert strategy.evaluate(user_msg, explicit_signal=True) is True

    # User message with signal = False -> False
    assert strategy.evaluate(user_msg, explicit_signal=False) is False

    # Assistant message with signal = True -> False
    assert strategy.evaluate(assistant_msg, explicit_signal=True) is False

    # System message with signal = True -> False
    assert strategy.evaluate(system_msg, explicit_signal=True) is False


@pytest.mark.asyncio
async def test_experience_promotion_service_promotes_user_message_with_user_id() -> None:
    """20, 21, 22. Verify ExperiencePromotionService creates Experience linked to user_id and source_message_id."""
    mock_record_exp = MagicMock(spec=RecordExperience)

    msg_id = uuid.uuid4()
    user_id = uuid.uuid4()
    raw_content = "I've decided to move to Bangalore."
    user_msg = Message(id=msg_id, role=MessageRole.USER, content=raw_content)

    expected_exp = Experience(
        id=uuid.uuid4(),
        content=raw_content,
        source=ExperienceSource.CHAT,
        user_id=str(user_id),
        source_message_id=msg_id,
        status=ExperienceStatus.RECEIVED,
    )
    mock_record_exp.execute = AsyncMock(return_value=expected_exp)

    service = ExperiencePromotionService(
        record_experience=mock_record_exp,
        strategy=DeterministicPromotionStrategy(),
    )

    result = await service.promote_message(user_msg, explicit_signal=True, user_id=user_id)

    assert isinstance(result, PromotionResult)
    assert result.promoted is True
    assert result.experience_id == expected_exp.id
    assert result.experience == expected_exp
    assert result.experience.user_id == str(user_id)
    assert result.experience.source_message_id == msg_id

    mock_record_exp.execute.assert_called_once_with(
        content=raw_content,
        source=ExperienceSource.CHAT,
        user_id=str(user_id),
        source_message_id=msg_id,
        type=None,
        domain=None,
        extraction_confidence=None,
    )


@pytest.mark.asyncio
async def test_extraction_failure_aborts_promotion_and_creates_no_experience() -> None:
    """PR #9 Requirements 1, 4 & 6B-G: Extraction failure MUST abort promotion and create NO Experience."""
    mock_record_exp = MagicMock(spec=RecordExperience)
    mock_record_exp.execute = AsyncMock()

    mock_extractor = MagicMock(spec=ExperienceExtractor)
    failed_extraction = ExperienceExtractionResult(
        success=False,
        content=None,
        domain=None,
        status=None,
        confidence=0.0,
        reasoning="Extraction LLM failure",
        raw_model="fallback_llm_error",
    )
    mock_extractor.extract = AsyncMock(return_value=failed_extraction)

    mock_strategy = MagicMock(spec=AIExperiencePromotionStrategy)
    classification = ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.9, confidence=0.95)
    mock_strategy.evaluate_async = AsyncMock(return_value=(True, classification))

    user_msg = Message(id=uuid.uuid4(), role=MessageRole.USER, content="Raw unpromoted text")

    service = ExperiencePromotionService(
        record_experience=mock_record_exp,
        strategy=mock_strategy,
        extractor=mock_extractor,
    )

    result = await service.promote_message(user_msg)

    # MUST NOT promote
    assert result.promoted is False
    assert result.experience_id is None
    # MUST NOT call RecordExperience.execute (raw message is NOT used as fallback)
    mock_record_exp.execute.assert_not_called()


@pytest.mark.asyncio
async def test_experience_promotion_service_duplicate_protection() -> None:
    """23 & B5. Verify duplicate promotion for same source_message_id is blocked."""
    mock_record_exp = MagicMock(spec=RecordExperience)
    mock_exp_repo = MagicMock(spec=ExperienceRepository)

    msg_id = uuid.uuid4()
    existing_exp = Experience(
        id=uuid.uuid4(),
        content="Already promoted content",
        source=ExperienceSource.CHAT,
        source_message_id=msg_id,
    )
    mock_exp_repo.get_by_source_message_id = AsyncMock(return_value=existing_exp)

    user_msg = Message(id=msg_id, role=MessageRole.USER, content="Duplicate prompt")
    service = ExperiencePromotionService(
        record_experience=mock_record_exp,
        strategy=DeterministicPromotionStrategy(),
        experience_repo=mock_exp_repo,
    )

    result = await service.promote_message(user_msg, explicit_signal=True)

    assert result.promoted is False
    assert result.experience == existing_exp
    mock_record_exp.execute.assert_not_called()


@pytest.mark.asyncio
async def test_experience_promotion_service_rejects_assistant_message() -> None:
    """Verify ExperiencePromotionService rejects assistant messages even with explicit signal."""
    mock_record_exp = MagicMock(spec=RecordExperience)
    mock_record_exp.execute = AsyncMock()

    assistant_msg = Message(id=uuid.uuid4(), role=MessageRole.ASSISTANT, content="Assistant text.")

    service = ExperiencePromotionService(
        record_experience=mock_record_exp,
        strategy=DeterministicPromotionStrategy(),
    )

    result = await service.promote_message(assistant_msg, explicit_signal=True)

    assert result.promoted is False
    assert result.experience_id is None
    mock_record_exp.execute.assert_not_called()
