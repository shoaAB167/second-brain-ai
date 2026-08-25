import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
from typing import List, Optional, Union
import uuid

from personal_ai.application.experience.extractor import ExperienceExtractor
from personal_ai.application.experience.record_experience import RecordExperience
from personal_ai.core.logger import get_logger
from personal_ai.db.models import Message
from personal_ai.domain.experience import (
    ClassificationResult,
    Experience,
    ExperienceExtractionResult,
    ExperienceRepository,
    ExperienceSource,
)
from personal_ai.llm.models import LLMMessage

logger = get_logger(__name__)


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

    Ensures ONLY user messages are promoted, provenance link (source_message_id) is established,
    duplicate promotions are blocked, classification records are linked via experience_id,
    and authenticated user ownership is enforced.
    """

    def __init__(
        self,
        record_experience: RecordExperience,
        strategy: Optional[PromotionStrategy] = None,
        experience_repo: Optional[ExperienceRepository] = None,
        extractor: Optional[ExperienceExtractor] = None,
    ) -> None:
        """Initialize ExperiencePromotionService.

        Args:
            record_experience: RecordExperience application use case.
            strategy: Optional PromotionStrategy implementation.
            experience_repo: Optional ExperienceRepository for duplicate protection checks.
            extractor: Optional ExperienceExtractor for structured experience extraction.
        """
        self._record_experience = record_experience
        self._strategy = strategy or DeterministicPromotionStrategy()
        self._experience_repo = experience_repo
        self._extractor = extractor

    async def promote_message(
        self,
        message: Message,
        explicit_signal: bool = False,
        user_id: Optional[Union[uuid.UUID, str]] = None,
        context: Optional[List[LLMMessage]] = None,
    ) -> PromotionResult:
        """Evaluate and conditionally promote a user Message to an Experience.

        Enforces duplicate protection (one message -> max 1 experience), classification linking, and user ownership.

        Args:
            message: Raw user Message entity.
            explicit_signal: Application control signal for deterministic promotion.
            user_id: Authenticated user UUID associated with the source conversation.
            context: Optional bounded prior conversation context for reference resolution.

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

        classification_res: Optional[ClassificationResult] = None
        if hasattr(self._strategy, "evaluate_async"):
            sig = inspect.signature(self._strategy.evaluate_async)
            if "context" in sig.parameters:
                should_promote, classification_res = await self._strategy.evaluate_async(message, context=context)
            else:
                should_promote, classification_res = await self._strategy.evaluate_async(message)
        else:
            should_promote = self._strategy.evaluate(message, explicit_signal=explicit_signal)

        # If classifier says is_experience=False or promotion policy rejects -> STOP
        if not should_promote:
            return PromotionResult(promoted=False)

        # Execute structured extraction ONLY IF classified as an experience and promotion policy passes
        extraction_res: Optional[ExperienceExtractionResult] = None
        if self._extractor and classification_res and classification_res.is_experience:
            try:
                extraction_res = await self._extractor.extract(
                    content=message.content,
                    classification=classification_res,
                    conversation_context=context,
                )
            except Exception as exc:
                logger.error("Extraction failed safely during message promotion: %s", exc)

        # Requirements 1, 4 & 5: If extractor is enabled, extraction MUST succeed (success == True).
        # If extraction fails or content is missing -> ABORT PROMOTION (DO NOT create Experience or fallback to raw content).
        if self._extractor:
            if not extraction_res or not extraction_res.success or not extraction_res.content:
                logger.warning(
                    "Experience promotion aborted because structured extraction failed or returned empty content [message_id=%s]",
                    message.id,
                )
                return PromotionResult(promoted=False)

        # Determine target content, canonical classification type, domain, and extraction_confidence to persist
        target_content = extraction_res.content if (extraction_res and extraction_res.success and extraction_res.content) else message.content
        target_type = classification_res.type if classification_res else None
        target_domain = extraction_res.domain if (extraction_res and extraction_res.success) else None
        target_confidence = extraction_res.confidence if (extraction_res and extraction_res.success) else None

        try:
            experience = await self._record_experience.execute(
                content=target_content,
                source=ExperienceSource.CHAT,
                user_id=str(user_id) if user_id else None,
                source_message_id=message.id,
                type=target_type,
                domain=target_domain,
                extraction_confidence=target_confidence,
            )
        except Exception:
            # Handle duplicate promotion race condition safely with retry loop
            if self._experience_repo and message.id:
                for _ in range(5):
                    existing = await self._experience_repo.get_by_source_message_id(message.id)
                    if existing:
                        return PromotionResult(
                            promoted=False,
                            experience_id=existing.id,
                            experience=existing,
                        )
                    await asyncio.sleep(0.02)
            raise

        # Link classification record experience_id to created Experience
        if hasattr(self._strategy, "update_classification_experience_id"):
            await self._strategy.update_classification_experience_id(experience.id)

        return PromotionResult(
            promoted=True,
            experience_id=experience.id,
            experience=experience,
        )
