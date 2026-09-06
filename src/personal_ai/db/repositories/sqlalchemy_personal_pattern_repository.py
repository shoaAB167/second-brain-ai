from datetime import datetime, timezone
from typing import List, Optional
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.db.models import PersonalPatternModel
from personal_ai.domain.pattern.entity import PersonalPattern
from personal_ai.domain.pattern.enums import PatternStatus
from personal_ai.domain.pattern.repository import PersonalPatternRepository


class SQLAlchemyPersonalPatternRepository(PersonalPatternRepository):
    """Concrete SQLAlchemy implementation of PersonalPatternRepository."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with SQLAlchemy AsyncSession.

        Args:
            session: Active database session.
        """
        self._session = session

    async def create(self, pattern: PersonalPattern) -> PersonalPattern:
        """Persist a new PersonalPattern entity."""
        first_obs = (
            pattern.first_observed_at.replace(tzinfo=timezone.utc)
            if pattern.first_observed_at.tzinfo is None
            else pattern.first_observed_at
        )
        last_obs = (
            pattern.last_observed_at.replace(tzinfo=timezone.utc)
            if pattern.last_observed_at.tzinfo is None
            else pattern.last_observed_at
        )
        created = (
            pattern.created_at.replace(tzinfo=timezone.utc)
            if pattern.created_at.tzinfo is None
            else pattern.created_at
        )
        updated = (
            pattern.updated_at.replace(tzinfo=timezone.utc)
            if pattern.updated_at.tzinfo is None
            else pattern.updated_at
        )

        model = PersonalPatternModel(
            id=pattern.id,
            user_id=pattern.user_id,
            description=pattern.description,
            domain=pattern.domain,
            evidence_ids=[str(eid) for eid in pattern.evidence_ids],
            confidence=pattern.confidence,
            status=pattern.status.value,
            first_observed_at=first_obs,
            last_observed_at=last_obs,
            superseded_by_id=pattern.superseded_by_id,
            created_at=created,
            updated_at=updated,
        )

        self._session.add(model)
        await self._session.commit()
        return self._model_to_domain(model)

    async def update(self, pattern: PersonalPattern) -> PersonalPattern:
        """Update an existing PersonalPattern entity scoped to user_id."""
        stmt = select(PersonalPatternModel).where(
            PersonalPatternModel.id == pattern.id,
            PersonalPatternModel.user_id == pattern.user_id,
        )
        res = await self._session.execute(stmt)
        model = res.scalar_one_or_none()
        if not model:
            raise ValueError(
                f"PersonalPattern with id {pattern.id} for user {pattern.user_id} not found for update."
            )

        model.description = pattern.description
        model.domain = pattern.domain
        model.evidence_ids = [str(eid) for eid in pattern.evidence_ids]
        model.confidence = pattern.confidence
        model.status = pattern.status.value
        model.first_observed_at = (
            pattern.first_observed_at.replace(tzinfo=timezone.utc)
            if pattern.first_observed_at.tzinfo is None
            else pattern.first_observed_at
        )
        model.last_observed_at = (
            pattern.last_observed_at.replace(tzinfo=timezone.utc)
            if pattern.last_observed_at.tzinfo is None
            else pattern.last_observed_at
        )
        model.superseded_by_id = pattern.superseded_by_id
        model.updated_at = (
            pattern.updated_at.replace(tzinfo=timezone.utc)
            if pattern.updated_at.tzinfo is None
            else pattern.updated_at
        )

        await self._session.commit()
        return self._model_to_domain(model)

    async def get_by_id(
        self,
        pattern_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
    ) -> Optional[PersonalPattern]:
        """Retrieve a pattern by its unique ID, optionally scoped to user_id."""
        stmt = select(PersonalPatternModel).where(PersonalPatternModel.id == pattern_id)
        if user_id is not None:
            stmt = stmt.where(PersonalPatternModel.user_id == user_id)
        res = await self._session.execute(stmt)
        model = res.scalar_one_or_none()
        return self._model_to_domain(model) if model else None

    async def list_by_user(
        self,
        user_id: uuid.UUID,
        status: Optional[PatternStatus] = None,
        domain: Optional[str] = None,
    ) -> List[PersonalPattern]:
        """List patterns belonging to a specific authenticated user with optional filtering."""
        stmt = select(PersonalPatternModel).where(PersonalPatternModel.user_id == user_id)

        if status is not None:
            status_str = status.value if hasattr(status, "value") else str(status)
            stmt = stmt.where(PersonalPatternModel.status == status_str)

        if domain is not None:
            dom_str = domain.value if hasattr(domain, "value") else str(domain)
            stmt = stmt.where(PersonalPatternModel.domain == dom_str.upper().strip())

        stmt = stmt.order_by(PersonalPatternModel.last_observed_at.desc())
        res = await self._session.execute(stmt)
        models = res.scalars().all()
        return [self._model_to_domain(m) for m in models]

    async def get_active_patterns(
        self,
        user_id: uuid.UUID,
        domain: Optional[str] = None,
    ) -> List[PersonalPattern]:
        """Retrieve active (HYPOTHESIS or CONFIRMED) patterns for a user."""
        stmt = (
            select(PersonalPatternModel)
            .where(
                PersonalPatternModel.user_id == user_id,
                PersonalPatternModel.status.in_([PatternStatus.HYPOTHESIS.value, PatternStatus.CONFIRMED.value]),
            )
        )
        if domain is not None:
            dom_str = domain.value if hasattr(domain, "value") else str(domain)
            stmt = stmt.where(PersonalPatternModel.domain == dom_str.upper().strip())

        stmt = stmt.order_by(PersonalPatternModel.confidence.desc(), PersonalPatternModel.last_observed_at.desc())
        res = await self._session.execute(stmt)
        models = res.scalars().all()
        return [self._model_to_domain(m) for m in models]

    @staticmethod
    def _model_to_domain(model: PersonalPatternModel) -> PersonalPattern:
        """Convert ORM model to domain entity."""
        try:
            status_enum = PatternStatus(model.status)
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid or corrupted PatternStatus '{model.status}' for pattern ID {model.id}."
            )

        evidence_list: List[uuid.UUID] = []
        if isinstance(model.evidence_ids, list):
            for eid in model.evidence_ids:
                if isinstance(eid, str):
                    try:
                        evidence_list.append(uuid.UUID(eid))
                    except ValueError:
                        continue
                elif isinstance(eid, uuid.UUID):
                    evidence_list.append(eid)

        return PersonalPattern(
            id=model.id,
            user_id=model.user_id,
            description=model.description,
            domain=model.domain,
            evidence_ids=evidence_list,
            confidence=float(model.confidence),
            status=status_enum,
            first_observed_at=model.first_observed_at,
            last_observed_at=model.last_observed_at,
            superseded_by_id=model.superseded_by_id,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
