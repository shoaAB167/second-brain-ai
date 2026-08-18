import asyncio
from typing import Optional, Union
import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
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

        Enforces source_message_id unique constraint race condition safety.

        Args:
            result: Structured classification metrics.
            source_message_id: Optional originating user Message UUID.
            experience_id: Optional promoted Experience UUID.

        Returns:
            ExperienceClassificationModel: Created or existing ORM model.
        """
        type_str = result.type.value if hasattr(result.type, "value") else (str(result.type) if result.type else None)
        source_msg_val = uuid.UUID(str(source_message_id)) if source_message_id else None
        exp_id_val = uuid.UUID(str(experience_id)) if experience_id else None

        model = ExperienceClassificationModel(
            source_message_id=source_msg_val,
            experience_id=exp_id_val,
            is_experience=result.is_experience,
            type=type_str,
            importance=result.importance,
            confidence=result.confidence,
            model=result.raw_model or "unknown",
        )

        try:
            self._session.add(model)
            await self._session.commit()
            return model
        except IntegrityError:
            await self._session.rollback()
            self._session.expire_all()
            if source_message_id:
                for _ in range(10):
                    existing = await self.get_by_source_message_id(source_message_id)
                    if existing:
                        return existing
                    await asyncio.sleep(0.05)
            raise

    async def get_by_source_message_id(
        self, source_message_id: Union[uuid.UUID, str]
    ) -> Optional[ExperienceClassificationModel]:
        """Retrieve an ExperienceClassificationModel entity by source message ID provenance.

        Args:
            source_message_id: UUID of the originating message.

        Returns:
            Optional[ExperienceClassificationModel]: Found model or None.
        """
        if not source_message_id:
            return None

        source_uuid = uuid.UUID(str(source_message_id)) if isinstance(source_message_id, (uuid.UUID, str)) else source_message_id

        stmt = select(ExperienceClassificationModel).where(
            ExperienceClassificationModel.source_message_id == source_uuid
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

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
            model.experience_id = uuid.UUID(str(experience_id))
            await self._session.commit()
