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
    ExperienceEvidenceLevel,
    ExperienceExtractionResult,
    ExperienceImportance,
    ExperienceLifecycle,
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
async def test_experience_promotion_service_promotes_user_message_when_extraction_disabled() -> None:
    """Requirement 5F: When extraction is disabled (extractor=None), existing promotion behavior remains unchanged."""
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
        importance=ExperienceImportance.MEDIUM,
        lifecycle=ExperienceLifecycle.STABLE,
        extraction_confidence=None,
        emotional_context=None,
        people_involved=None,
        temporal_context=None,
        evidence_level="EXTRACTED",
    )


@pytest.mark.asyncio
async def test_extraction_succeeds_creates_experience_with_structured_content_and_classifier_type() -> None:
    """Requirement 5A & 5G: Successful extraction creates Experience with structured content and canonical classifier type."""
    mock_record_exp = MagicMock(spec=RecordExperience)
    exp_id = uuid.uuid4()
    msg_id = uuid.uuid4()
    user_id = uuid.uuid4()

    mock_record_exp.execute = AsyncMock(
        return_value=Experience(
            id=exp_id,
            content="Reach 30 LPA as backend engineer",
            type=ExperienceType.GOAL,
            domain="career",
            extraction_confidence=0.95,
            source=ExperienceSource.CHAT,
            user_id=str(user_id),
            source_message_id=msg_id,
            status=ExperienceStatus.RECEIVED,
        )
    )

    mock_extractor = MagicMock(spec=ExperienceExtractor)
    successful_extraction = ExperienceExtractionResult(
        success=True,
        content="Reach 30 LPA as backend engineer",
        domain="career",
        status="active",
        confidence=0.95,
        reasoning="Goal extracted",
        raw_model="gpt-4o-mini",
    )
    mock_extractor.extract = AsyncMock(return_value=successful_extraction)

    mock_strategy = MagicMock(spec=AIExperiencePromotionStrategy)
    classification = ClassificationResult(is_experience=True, type=ExperienceType.GOAL, importance=0.9, confidence=0.95)
    mock_strategy.evaluate_async = AsyncMock(return_value=(True, classification))

    user_msg = Message(id=msg_id, role=MessageRole.USER, content="I want to reach 30 LPA as a backend engineer.")

    service = ExperiencePromotionService(
        record_experience=mock_record_exp,
        strategy=mock_strategy,
        extractor=mock_extractor,
    )

    result = await service.promote_message(user_msg, user_id=user_id)

    assert result.promoted is True
    assert result.experience_id == exp_id
    mock_record_exp.execute.assert_called_once_with(
        content="Reach 30 LPA as backend engineer",
        source=ExperienceSource.CHAT,
        user_id=str(user_id),
        source_message_id=msg_id,
        type=ExperienceType.GOAL,  # Canonical classifier type preserved!
        domain="career",
        importance=ExperienceImportance.MEDIUM,
        lifecycle=ExperienceLifecycle.STABLE,
        extraction_confidence=0.95,
        emotional_context=None,
        people_involved=None,
        temporal_context=None,
        evidence_level=ExperienceEvidenceLevel.EXTRACTED,
    )


@pytest.mark.asyncio
async def test_extraction_failure_aborts_promotion_and_creates_no_experience() -> None:
    """Requirements 5B, 5C, 5D, 5E & 5H: Extraction failure MUST abort promotion and create NO Experience (raw message NOT stored)."""
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
async def test_extraction_exception_handled_safely_creates_no_experience() -> None:
    """Requirement 5E: Unexpected exception during extractor.extract creates NO Experience and remains safe."""
    mock_record_exp = MagicMock(spec=RecordExperience)
    mock_record_exp.execute = AsyncMock()

    mock_extractor = MagicMock(spec=ExperienceExtractor)
    mock_extractor.extract = AsyncMock(side_effect=RuntimeError("Extractor crashed unexpectedly"))

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

    assert result.promoted is False
    assert result.experience_id is None
    mock_record_exp.execute.assert_not_called()


@pytest.mark.asyncio
async def test_experience_promotion_service_duplicate_protection() -> None:
    """Requirement 6: Verify duplicate promotion for same source_message_id is blocked."""
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
