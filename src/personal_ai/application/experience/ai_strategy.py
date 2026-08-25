from typing import List, Optional
import uuid

from personal_ai.application.experience.classifier import ExperienceClassifier
from personal_ai.application.experience.promotion import PromotionStrategy
from personal_ai.config.settings import get_settings
from personal_ai.core.logger import get_logger
from personal_ai.db.models import ExperienceClassificationModel, Message
from personal_ai.db.repositories.sqlalchemy_experience_classification_repository import (
    SQLAlchemyExperienceClassificationRepository,
)
from personal_ai.domain.experience import ClassificationResult
from personal_ai.llm.models import LLMMessage

logger = get_logger(__name__)


class AIExperiencePromotionStrategy(PromotionStrategy):
    """AI-powered promotion strategy evaluating user messages with ExperienceClassifier.

    Applies application promotion thresholds (min_confidence and min_importance)
    and saves provenance records to the experience_classifications repository.
    """

    def __init__(
        self,
        classifier: ExperienceClassifier,
        min_confidence: Optional[float] = None,
        min_importance: Optional[float] = None,
        classification_repo: Optional[SQLAlchemyExperienceClassificationRepository] = None,
    ) -> None:
        """Initialize strategy with classifier, thresholds, and optional repository.

        Args:
            classifier: ExperienceClassifier instance.
            min_confidence: Minimum confidence threshold (defaults to Settings.experience_classifier_min_confidence).
            min_importance: Minimum importance threshold (defaults to Settings.experience_classifier_min_importance).
            classification_repo: Optional repository for persisting classification records.
        """
        settings = get_settings()
        self._classifier = classifier
        self._min_confidence = (
            min_confidence
            if min_confidence is not None
            else settings.experience_classifier_min_confidence
        )
        self._min_importance = (
            min_importance
            if min_importance is not None
            else settings.experience_classifier_min_importance
        )
        self._classification_repo = classification_repo
        self._last_classification_model: Optional[ExperienceClassificationModel] = None

    def evaluate(self, message: Message, explicit_signal: bool = False) -> bool:
        """Sync fallback evaluation signature for PromotionStrategy interface compatibility."""
        role_str = message.role.value if hasattr(message.role, "value") else str(message.role)
        if role_str.lower() != "user":
            return False
        return bool(explicit_signal)

    async def evaluate_async(
        self,
        message: Message,
        context: Optional[List[LLMMessage]] = None,
    ) -> tuple[bool, Optional[ClassificationResult]]:
        """Asynchronously classify user message and evaluate application promotion policy.

        Args:
            message: Raw user Message entity.
            context: Optional list of recent LLMMessage objects for contextual reference resolution.

        Returns:
            tuple[bool, Optional[ClassificationResult]]: (should_promote, classification_result)
        """
        role_str = message.role.value if hasattr(message.role, "value") else str(message.role)
        if role_str.lower() != "user":
            return False, None

        result = await self._classifier.classify(message.content, conversation_context=context)

        # Save classification provenance record to database if repository is provided
        if self._classification_repo:
            try:
                self._last_classification_model = await self._classification_repo.create(
                    result=result,
                    source_message_id=message.id,
                )
            except Exception as exc:
                logger.error("Failed to persist classification provenance record: %s", exc)

        # Application promotion policy decision
        should_promote = (
            result.is_experience
            and (result.confidence >= self._min_confidence)
            and (result.importance >= self._min_importance)
        )

        logger.info(
            "Promotion policy evaluated [should_promote=%s, is_exp=%s, conf=%.2f (>=%.2f), imp=%.2f (>=%.2f)]",
            should_promote,
            result.is_experience,
            result.confidence,
            self._min_confidence,
            result.importance,
            self._min_importance,
        )

        return should_promote, result

    async def update_classification_experience_id(self, experience_id: uuid.UUID) -> None:
        """Link the last created classification record to the promoted Experience ID.

        Args:
            experience_id: Promoted Experience UUID.
        """
        if self._classification_repo and self._last_classification_model:
            try:
                await self._classification_repo.update_experience_id(
                    classification_id=self._last_classification_model.id,
                    experience_id=experience_id,
                )
                self._last_classification_model.experience_id = experience_id
            except Exception as exc:
                logger.error("Failed to link classification record to experience_id: %s", exc)
