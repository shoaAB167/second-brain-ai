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
from personal_ai.application.experience.embedding_service import ExperienceEmbeddingService
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
from personal_ai.domain.experience.entity import Experience
from personal_ai.infrastructure.embedding import EmbeddingProvider, get_embedding_provider
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
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        """Initialize background experience processor.

        Args:
            session_factory: SQLAlchemy async_sessionmaker for isolated session spawning.
            llm_client: LLMClient for AI classification.
            strategy: Optional PromotionStrategy override for tests or custom policies.
            extractor: Optional ExperienceExtractor for structured experience extraction.
            embedding_provider: Optional EmbeddingProvider override for tests or custom models.
        """
        self._session_factory = session_factory
        self._llm_client = llm_client
        self._strategy = strategy
        self._extractor = extractor
        self._embedding_provider = embedding_provider

    async def process_background_promotion(
        self,
        message: Message,
        user_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Execute background classification, promotion, and vector embedding in separated DB transactions."""
        settings = get_settings()
        promoted_experience: Optional[Experience] = None

        # TRANSACTION 1: Classification, extraction, and initial Experience persistence
        try:
            async with self._session_factory() as bg_session:
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
                extractor = self._extractor
                if not strategy:
                    classifier = ExperienceClassifier(llm_client=self._llm_client)
                    strategy = AIExperiencePromotionStrategy(
                        classifier=classifier,
                        classification_repo=classification_repo,
                    )
                    if not extractor:
                        extractor = ExperienceExtractor(llm_client=self._llm_client)

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
                if res.promoted and res.experience:
                    promoted_experience = res.experience
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
            return

        # Transaction 1 closed!

        # EXTERNAL EMBEDDING API CALL (Outside DB transaction):
        if promoted_experience and settings.embedding_enabled:
            provider = self._embedding_provider or get_embedding_provider()
            embedding_service = ExperienceEmbeddingService(provider=provider)
            embed_res = await embedding_service.embed_experience(promoted_experience)

            # TRANSACTION 2: Update Experience embedding & status in fresh DB transaction
            try:
                async with self._session_factory() as update_session:
                    update_repo = SQLAlchemyExperienceRepository(session=update_session)
                    existing_exp = await update_repo.get_by_id(promoted_experience.id)
                    if existing_exp:
                        existing_exp.embedding = embed_res.embedding if embed_res.success else None
                        existing_exp.embedding_model = embed_res.embedding_model if embed_res.success else None
                        existing_exp.embedding_status = embed_res.status
                        existing_exp.embedded_at = embed_res.embedded_at

                        await update_repo.update(existing_exp)

                        if embed_res.success:
                            logger.info(
                                "Experience vector embedding completed [experience_id=%s, model=%s]",
                                existing_exp.id,
                                embed_res.embedding_model,
                            )
                        else:
                            logger.warning(
                                "Experience vector embedding failed safely [experience_id=%s]: %s",
                                existing_exp.id,
                                embed_res.error,
                            )
            except Exception as update_exc:
                logger.error(
                    "Unexpected error persisting vector embedding update [experience_id=%s]: %s",
                    promoted_experience.id,
                    update_exc,
                )
