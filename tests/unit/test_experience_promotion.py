import uuid
from unittest.mock import AsyncMock, MagicMock
import pytest

from personal_ai.application.experience import (
    DeterministicPromotionStrategy,
    ExperiencePromotionService,
    PromotionResult,
    RecordExperience,
)
from personal_ai.db.models import Message, MessageRole
from personal_ai.domain.experience import Experience, ExperienceSource, ExperienceStatus


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

    # Assistant message with signal = True -> False (Never eligible)
    assert strategy.evaluate(assistant_msg, explicit_signal=True) is False

    # System message with signal = True -> False (Never eligible)
    assert strategy.evaluate(system_msg, explicit_signal=True) is False


@pytest.mark.asyncio
async def test_experience_promotion_service_promotes_user_message() -> None:
    """Verify ExperiencePromotionService creates Experience linked to source_message_id for user message."""
    mock_record_exp = MagicMock(spec=RecordExperience)

    msg_id = uuid.uuid4()
    raw_content = "I've decided to move to Bangalore."
    user_msg = Message(id=msg_id, role=MessageRole.USER, content=raw_content)

    expected_exp = Experience(
        id=uuid.uuid4(),
        content=raw_content,
        source=ExperienceSource.CHAT,
        source_message_id=msg_id,
        status=ExperienceStatus.RECEIVED,
    )
    mock_record_exp.execute = AsyncMock(return_value=expected_exp)

    service = ExperiencePromotionService(
        record_experience=mock_record_exp,
        strategy=DeterministicPromotionStrategy(),
    )

    result = await service.promote_message(user_msg, explicit_signal=True)

    assert isinstance(result, PromotionResult)
    assert result.promoted is True
    assert result.experience_id == expected_exp.id
    assert result.experience == expected_exp

    mock_record_exp.execute.assert_called_once_with(
        content=raw_content,
        source=ExperienceSource.CHAT,
        user_id=None,
        source_message_id=msg_id,
    )


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
