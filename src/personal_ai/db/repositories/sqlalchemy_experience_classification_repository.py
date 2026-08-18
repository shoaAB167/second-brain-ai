from typing import Optional
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.db.models import ExperienceClassificationModel
from personal_ai.domain.experience import ClassificationResult


class SQLAlchemyExperienceClassificationRepository:
    """Concrete SQLAlchemy implementation for storing and managing ExperienceClassificationModel records."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with SQLAlchemy AsyncSession.

        Args:
            session: Active database session.
        """
        self._session = session

    async def create(
        self,
        result: ClassificationResult,
        source_message_id: Optional[uuid.UUID] = None,
        experience_id: Optional[uuid.UUID] = None,
    ) -> ExperienceClassificationModel:
        """Persist a new ExperienceClassificationModel to the database.

        Args:
            result: Structured classification metrics.
            source_message_id: Optional originating user Message UUID.
            experience_id: Optional promoted Experience UUID.

        Returns:
            ExperienceClassificationModel: Created ORM model.
        """
        type_str = result.type.value if hasattr(result.type, "value") else (str(result.type) if result.type else None)

        model = ExperienceClassificationModel(
            source_message_id=source_message_id,
            experience_id=experience_id,
            is_experience=result.is_experience,
            type=type_str,
            importance=result.importance,
            confidence=result.confidence,
            model=result.raw_model or "unknown",
        )

        self._session.add(model)
        await self._session.commit()
        return model

    async def update_experience_id(
        self,
        classification_id: uuid.UUID,
        experience_id: uuid.UUID,
    ) -> None:
        """Link an existing classification record to a promoted Experience ID.

        Args:
            classification_id: UUID of classification record.
            experience_id: UUID of promoted Experience entity.
        """
        stmt = select(ExperienceClassificationModel).where(
            ExperienceClassificationModel.id == classification_id
        )
        res = await self._session.execute(stmt)
        model = res.scalar_one_or_none()
        if model:
            model.experience_id = experience_id
            await self._session.commit()
