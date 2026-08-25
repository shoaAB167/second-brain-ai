from typing import List, Optional
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from personal_ai.application.experience.ai_strategy import (
    AIExperiencePromotionStrategy,
)
from personal_ai.application.experience.background_processor import (
    BackgroundExperienceProcessor,
)
from personal_ai.application.experience.classifier import ExperienceClassifier
from personal_ai.application.experience.extractor import ExperienceExtractor
from personal_ai.application.experience.promotion import (
    ExperiencePromotionService,
    PromotionStrategy,
    RecordExperience,
)
from personal_ai.config.settings import get_settings
from personal_ai.core.logger import get_logger
from personal_ai.db.models import Message
from personal_ai.db.repositories.sqlalchemy_experience_classification_repository import (
    SQLAlchemyExperienceClassificationRepository,
)
from personal_ai.db.repositories.sqlalchemy_experience_repository import (
    SQLAlchemyExperienceRepository,
)
from personal_ai.llm.client import LLMClient
from personal_ai.llm.models import LLMMessage

logger = get_logger(__name__)


class SQLAlchemyBackgroundExperienceProcessor(BackgroundExperienceProcessor):
    """Concrete implementation of BackgroundExperienceProcessor using isolated AsyncSession instances."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        llm_client: LLMClient,
        strategy: Optional[PromotionStrategy] = None,
        extractor: Optional[ExperienceExtractor] = None,
    ) -> None:
        """Initialize background experience processor.

        Args:
            session_factory: SQLAlchemy async_sessionmaker for isolated session spawning.
            llm_client: LLMClient for AI classification.
            strategy: Optional PromotionStrategy override for tests or custom policies.
            extractor: Optional ExperienceExtractor for structured experience extraction.
        """
        self._session_factory = session_factory
        self._llm_client = llm_client
        self._strategy = strategy
        self._extractor = extractor

    async def process_background_promotion(
        self,
        message: Message,
        user_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Execute background classification and promotion in an isolated DB transaction."""
        settings = get_settings()
        try:
            async with self._session_factory() as bg_session:
                # Retrieve bounded prior conversation messages for reference resolution
                context_messages: List[LLMMessage] = []
                limit = settings.experience_classifier_context_messages
                if message.conversation_id and limit > 0:
                    stmt = (
                        select(Message)
                        .where(
                            Message.conversation_id == message.conversation_id,
                            Message.id != message.id,
                            Message.created_at <= message.created_at,
                        )
                        .order_by(Message.created_at.desc())
                        .limit(limit)
                    )
                    res = await bg_session.execute(stmt)
                    prior_msgs = list(res.scalars().all())
                    prior_msgs.reverse()
                    context_messages = [
                        LLMMessage(
                            role=m.role.value if hasattr(m.role, "value") else str(m.role),
                            content=m.content,
                        )
                        for m in prior_msgs
                    ]

                exp_repo = SQLAlchemyExperienceRepository(session=bg_session)
                record_exp = RecordExperience(repository=exp_repo)
                classification_repo = SQLAlchemyExperienceClassificationRepository(
                    session=bg_session
                )

                strategy = self._strategy
                if not strategy:
                    classifier = ExperienceClassifier(llm_client=self._llm_client)
                    strategy = AIExperiencePromotionStrategy(
                        classifier=classifier,
                        classification_repo=classification_repo,
                    )

                extractor = self._extractor or ExperienceExtractor(llm_client=self._llm_client)

                service = ExperiencePromotionService(
                    record_experience=record_exp,
                    strategy=strategy,
                    experience_repo=exp_repo,
                    extractor=extractor,
                )

                res = await service.promote_message(
                    message=message,
                    user_id=user_id,
                    context=context_messages,
                )
                if res.promoted:
                    logger.info(
                        "Background experience promoted successfully [message_id=%s, experience_id=%s, user_id=%s]",
                        message.id,
                        res.experience_id,
                        user_id,
                    )
        except Exception as exc:
            logger.error(
                "Background experience promotion failed safely [message_id=%s, user_id=%s]: %s",
                message.id,
                user_id,
                exc,
            )
