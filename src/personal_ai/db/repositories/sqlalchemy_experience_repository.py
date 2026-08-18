import asyncio
from typing import Optional, Union
import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.db.models import ExperienceModel
from personal_ai.domain.experience import (
    Experience,
    ExperienceRepository,
    ExperienceSource,
    ExperienceStatus,
)


class SQLAlchemyExperienceRepository(ExperienceRepository):
    """Concrete SQLAlchemy implementation of ExperienceRepository interface."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with SQLAlchemy AsyncSession.

        Args:
            session: Active database session.
        """
        self._session = session

    async def create(self, experience: Experience) -> Experience:
        """Persist a new Experience entity to PostgreSQL.

        Enforces source_message_id unique constraint race condition safety.

        Args:
            experience: Domain Experience entity to persist.

        Returns:
            Experience: Persisted Experience domain entity (or existing on duplicate race).
        """
        user_id_val = uuid.UUID(str(experience.user_id)) if experience.user_id else None
        source_msg_val = uuid.UUID(str(experience.source_message_id)) if experience.source_message_id else None

        model = ExperienceModel(
            id=experience.id,
            user_id=user_id_val,
            source_message_id=source_msg_val,
            content=experience.content,
            source=experience.source.value if hasattr(experience.source, "value") else str(experience.source),
            status=experience.status.value if hasattr(experience.status, "value") else str(experience.status),
            created_at=experience.created_at,
        )

        try:
            self._session.add(model)
            await self._session.commit()
            return self._model_to_domain(model)
        except IntegrityError:
            await self._session.rollback()
            self._session.expire_all()
            if experience.source_message_id:
                for _ in range(10):
                    existing = await self.get_by_source_message_id(experience.source_message_id)
                    if existing:
                        return existing
                    await asyncio.sleep(0.05)
            raise

    async def get_by_id(self, experience_id: uuid.UUID) -> Optional[Experience]:
        """Retrieve an Experience entity by UUID.

        Args:
            experience_id: UUID of the target experience.

        Returns:
            Optional[Experience]: Found domain entity or None.
        """
        stmt = select(ExperienceModel).where(ExperienceModel.id == experience_id)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()

        if not model:
            return None

        return self._model_to_domain(model)

    async def get_by_source_message_id(self, source_message_id: Union[uuid.UUID, str]) -> Optional[Experience]:
        """Retrieve an Experience domain entity by source message ID provenance.

        Args:
            source_message_id: UUID of the originating message.

        Returns:
            Optional[Experience]: Found domain entity or None.
        """
        if not source_message_id:
            return None

        source_uuid = uuid.UUID(str(source_message_id)) if isinstance(source_message_id, (uuid.UUID, str)) else source_message_id

        stmt = select(ExperienceModel).where(ExperienceModel.source_message_id == source_uuid)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()

        if not model:
            return None

        return self._model_to_domain(model)

    @staticmethod
    def _model_to_domain(model: ExperienceModel) -> Experience:
        """Convert SQLAlchemy ExperienceModel to domain Experience entity."""
        return Experience(
            id=model.id,
            user_id=model.user_id,
            source_message_id=model.source_message_id,
            content=model.content,
            source=ExperienceSource(model.source),
            status=ExperienceStatus(model.status),
            created_at=model.created_at,
        )
