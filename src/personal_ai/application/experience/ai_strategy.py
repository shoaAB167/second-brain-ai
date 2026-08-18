from typing import Optional

from personal_ai.application.experience.classifier import ExperienceClassifier
from personal_ai.application.experience.promotion import PromotionStrategy
from personal_ai.config.settings import get_settings
from personal_ai.core.logger import get_logger
from personal_ai.db.models import Message
from personal_ai.db.repositories.sqlalchemy_experience_classification_repository import (
    SQLAlchemyExperienceClassificationRepository,
)
from personal_ai.domain.experience import ClassificationResult

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

    def evaluate(self, message: Message, explicit_signal: bool = False) -> bool:
        """Sync fallback evaluation signature for PromotionStrategy interface compatibility."""
        role_str = message.role.value if hasattr(message.role, "value") else str(message.role)
        if role_str.lower() != "user":
            return False
        return bool(explicit_signal)

    async def evaluate_async(self, message: Message) -> tuple[bool, Optional[ClassificationResult]]:
        """Asynchronously classify user message and evaluate application promotion policy.

        Args:
            message: Raw user Message entity.

        Returns:
            tuple[bool, Optional[ClassificationResult]]: (should_promote, classification_result)
        """
        role_str = message.role.value if hasattr(message.role, "value") else str(message.role)
        if role_str.lower() != "user":
            return False, None

        result = await self._classifier.classify(message.content)

        # Save classification provenance record to database if repository is provided
        if self._classification_repo:
            try:
                await self._classification_repo.create(
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
