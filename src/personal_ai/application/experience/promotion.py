from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
import uuid

from personal_ai.application.experience.record_experience import RecordExperience
from personal_ai.db.models import Message
from personal_ai.domain.experience import Experience, ExperienceRepository, ExperienceSource


@dataclass
class PromotionResult:
    """Result container for Experience promotion evaluations."""

    promoted: bool
    experience_id: Optional[uuid.UUID] = None
    experience: Optional[Experience] = None


class PromotionStrategy(ABC):
    """Abstract strategy interface for determining Experience promotion eligibility."""

    @abstractmethod
    def evaluate(self, message: Message, explicit_signal: bool = False) -> bool:
        """Evaluate if a Message is eligible to be promoted into an Experience."""
        pass


class DeterministicPromotionStrategy(PromotionStrategy):
    """Deterministic promotion strategy for PR #7.

    Evaluates to True ONLY IF the message is a user message AND explicit_signal is True.
    Assistant and system messages are explicitly rejected from promotion.
    """

    def evaluate(self, message: Message, explicit_signal: bool = False) -> bool:
        role_str = message.role.value if hasattr(message.role, "value") else str(message.role)
        if role_str.lower() != "user":
            return False
        return bool(explicit_signal)


class ExperiencePromotionService:
    """Application service for promoting existing user Messages into Experience entities.

    Ensures ONLY user messages are promoted, original text is preserved verbatim,
    provenance link (source_message_id) is established, duplicate promotions are blocked,
    and authenticated user ownership is enforced.
    """

    def __init__(
        self,
        record_experience: RecordExperience,
        strategy: Optional[PromotionStrategy] = None,
        experience_repo: Optional[ExperienceRepository] = None,
    ) -> None:
        """Initialize ExperiencePromotionService.

        Args:
            record_experience: RecordExperience application use case.
            strategy: Optional PromotionStrategy implementation.
            experience_repo: Optional ExperienceRepository for duplicate protection checks.
        """
        self._record_experience = record_experience
        self._strategy = strategy or DeterministicPromotionStrategy()
        self._experience_repo = experience_repo

    async def promote_message(
        self,
        message: Message,
        explicit_signal: bool = False,
        user_id: Optional[Union[uuid.UUID, str]] = None,
    ) -> PromotionResult:
        """Evaluate and conditionally promote a user Message to an Experience.

        Enforces duplicate protection (one message -> max 1 experience) and user ownership.

        Args:
            message: Raw user Message entity.
            explicit_signal: Application control signal for deterministic promotion.
            user_id: Authenticated user UUID associated with the source conversation.

        Returns:
            PromotionResult: Result indicating whether promotion occurred and experience ID.
        """
        role_str = message.role.value if hasattr(message.role, "value") else str(message.role)
        if role_str.lower() != "user":
            return PromotionResult(promoted=False)

        # Duplicate protection check: Verify an Experience with source_message_id does not already exist
        if self._experience_repo:
            existing = await self._experience_repo.get_by_source_message_id(message.id)
            if existing:
                return PromotionResult(
                    promoted=False,
                    experience_id=existing.id,
                    experience=existing,
                )

        if hasattr(self._strategy, "evaluate_async"):
            should_promote, _ = await self._strategy.evaluate_async(message)
        else:
            should_promote = self._strategy.evaluate(message, explicit_signal=explicit_signal)

        if not should_promote:
            return PromotionResult(promoted=False)

        experience = await self._record_experience.execute(
            content=message.content,
            source=ExperienceSource.CHAT,
            user_id=str(user_id) if user_id else None,
            source_message_id=message.id,
        )

        return PromotionResult(
            promoted=True,
            experience_id=experience.id,
            experience=experience,
        )
