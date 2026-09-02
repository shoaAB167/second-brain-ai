from typing import List, Optional
import uuid

from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.db.models import ExperienceRelationshipModel
from personal_ai.domain.experience.enums import ExperienceRelationshipType
from personal_ai.domain.experience.relationship import ExperienceRelationship
from personal_ai.domain.experience.relationship_repository import ExperienceRelationshipRepository


class SQLAlchemyExperienceRelationshipRepository(ExperienceRelationshipRepository):
    """Concrete SQLAlchemy implementation of ExperienceRelationshipRepository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with SQLAlchemy AsyncSession.

        Args:
            session: Active database session.
        """
        self._session = session

    async def create(self, relationship: ExperienceRelationship) -> ExperienceRelationship:
        """Persist a new ExperienceRelationship entity.

        Args:
            relationship: Domain relationship entity to persist.

        Returns:
            ExperienceRelationship: Persisted domain entity.
        """
        rel_type_str = (
            relationship.relationship_type.value
            if hasattr(relationship.relationship_type, "value")
            else str(relationship.relationship_type)
        )

        model = ExperienceRelationshipModel(
            id=relationship.id,
            source_experience_id=relationship.source_experience_id,
            target_experience_id=relationship.target_experience_id,
            relationship_type=rel_type_str,
            confidence=relationship.confidence,
            reason=relationship.reason,
            created_at=relationship.created_at,
        )

        try:
            self._session.add(model)
            await self._session.commit()
            return self._model_to_domain(model)
        except IntegrityError:
            await self._session.rollback()
            # If relationship already exists, return existing
            stmt = select(ExperienceRelationshipModel).where(
                ExperienceRelationshipModel.source_experience_id == relationship.source_experience_id,
                ExperienceRelationshipModel.target_experience_id == relationship.target_experience_id,
                ExperienceRelationshipModel.relationship_type == rel_type_str,
            )
            res = await self._session.execute(stmt)
            existing = res.scalar_one_or_none()
            if existing:
                return self._model_to_domain(existing)
            raise

    async def get_by_source_id(self, source_id: uuid.UUID) -> List[ExperienceRelationship]:
        """Retrieve all relationships where the given experience is the source."""
        stmt = (
            select(ExperienceRelationshipModel)
            .where(ExperienceRelationshipModel.source_experience_id == source_id)
            .order_by(ExperienceRelationshipModel.created_at.asc())
        )
        res = await self._session.execute(stmt)
        models = res.scalars().all()
        return [self._model_to_domain(m) for m in models]

    async def get_by_target_id(self, target_id: uuid.UUID) -> List[ExperienceRelationship]:
        """Retrieve all relationships where the given experience is the target."""
        stmt = (
            select(ExperienceRelationshipModel)
            .where(ExperienceRelationshipModel.target_experience_id == target_id)
            .order_by(ExperienceRelationshipModel.created_at.asc())
        )
        res = await self._session.execute(stmt)
        models = res.scalars().all()
        return [self._model_to_domain(m) for m in models]

    async def get_by_experience_id(self, experience_id: uuid.UUID) -> List[ExperienceRelationship]:
        """Retrieve all relationships where the given experience is either source or target."""
        stmt = (
            select(ExperienceRelationshipModel)
            .where(
                or_(
                    ExperienceRelationshipModel.source_experience_id == experience_id,
                    ExperienceRelationshipModel.target_experience_id == experience_id,
                )
            )
            .order_by(ExperienceRelationshipModel.created_at.asc())
        )
        res = await self._session.execute(stmt)
        models = res.scalars().all()
        return [self._model_to_domain(m) for m in models]

    async def exists(
        self,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
        relationship_type: ExperienceRelationshipType,
    ) -> bool:
        """Check if a specific relationship already exists."""
        rel_type_str = (
            relationship_type.value
            if hasattr(relationship_type, "value")
            else str(relationship_type)
        )
        stmt = select(ExperienceRelationshipModel.id).where(
            ExperienceRelationshipModel.source_experience_id == source_id,
            ExperienceRelationshipModel.target_experience_id == target_id,
            ExperienceRelationshipModel.relationship_type == rel_type_str,
        )
        res = await self._session.execute(stmt)
        return res.scalar_one_or_none() is not None

    @staticmethod
    def _model_to_domain(model: ExperienceRelationshipModel) -> ExperienceRelationship:
        """Convert ORM model to domain entity."""
        try:
            rel_type = ExperienceRelationshipType(model.relationship_type)
        except ValueError:
            rel_type = ExperienceRelationshipType.RELATED

        return ExperienceRelationship(
            id=model.id,
            source_experience_id=model.source_experience_id,
            target_experience_id=model.target_experience_id,
            relationship_type=rel_type,
            confidence=float(model.confidence),
            reason=model.reason,
            created_at=model.created_at,
        )
