from datetime import datetime, timezone
from typing import List, Optional
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.db.models import PersonModel
from personal_ai.domain.person.entity import Person
from personal_ai.domain.person.enums import RelationshipType
from personal_ai.domain.person.repository import PersonRepository


class SQLAlchemyPersonRepository(PersonRepository):
    """Concrete SQLAlchemy implementation of PersonRepository (PR #29).

    Enforces strict user isolation across all operations.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with active SQLAlchemy AsyncSession.

        Args:
            session: Active database session.
        """
        self._session = session

    def _model_to_domain(self, model: PersonModel) -> Person:
        """Map ORM PersonModel entity to pure domain Person dataclass."""
        rel_type = RelationshipType.OTHER
        if model.relationship_type:
            try:
                rel_type = RelationshipType(model.relationship_type.upper().strip())
            except ValueError:
                rel_type = RelationshipType.OTHER

        return Person(
            id=model.id,
            user_id=model.user_id,
            name=model.name,
            relationship_type=rel_type,
            notes=model.notes,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    async def create(self, person: Person) -> Person:
        """Persist a new Person entity."""
        created = (
            person.created_at.replace(tzinfo=timezone.utc)
            if person.created_at.tzinfo is None
            else person.created_at
        )
        updated = (
            person.updated_at.replace(tzinfo=timezone.utc)
            if person.updated_at.tzinfo is None
            else person.updated_at
        )

        model = PersonModel(
            id=person.id,
            user_id=person.user_id,
            name=person.name,
            relationship_type=person.relationship_type.value,
            notes=person.notes,
            created_at=created,
            updated_at=updated,
        )

        self._session.add(model)
        await self._session.commit()
        return self._model_to_domain(model)

    async def get_by_id(
        self,
        user_id: uuid.UUID,
        person_id: uuid.UUID,
    ) -> Optional[Person]:
        """Retrieve a Person by its unique ID, strictly scoped to authenticated user_id."""
        stmt = select(PersonModel).where(
            PersonModel.id == person_id,
            PersonModel.user_id == user_id,
        )
        res = await self._session.execute(stmt)
        model = res.scalar_one_or_none()
        return self._model_to_domain(model) if model else None

    async def list(
        self,
        user_id: uuid.UUID,
    ) -> List[Person]:
        """List all Person records belonging to a specific authenticated user."""
        stmt = (
            select(PersonModel)
            .where(PersonModel.user_id == user_id)
            .order_by(PersonModel.name.asc())
        )
        res = await self._session.execute(stmt)
        models = res.scalars().all()
        return [self._model_to_domain(m) for m in models]

    async def find_by_name(
        self,
        user_id: uuid.UUID,
        name: str,
    ) -> Optional[Person]:
        """Find a Person by case-insensitive exact name match scoped to user_id."""
        if not name or not name.strip():
            return None

        clean_name = name.strip().lower()
        stmt = select(PersonModel).where(
            PersonModel.user_id == user_id,
            func.lower(PersonModel.name) == clean_name,
        )
        res = await self._session.execute(stmt)
        model = res.scalar_one_or_none()
        return self._model_to_domain(model) if model else None

    async def update(
        self,
        user_id: uuid.UUID,
        person: Person,
    ) -> Person:
        """Update an existing Person entity scoped to user_id."""
        stmt = select(PersonModel).where(
            PersonModel.id == person.id,
            PersonModel.user_id == user_id,
        )
        res = await self._session.execute(stmt)
        model = res.scalar_one_or_none()
        if not model:
            raise ValueError(
                f"Person with id {person.id} for user {user_id} not found for update."
            )

        model.name = person.name
        model.relationship_type = person.relationship_type.value
        model.notes = person.notes
        model.updated_at = (
            person.updated_at.replace(tzinfo=timezone.utc)
            if person.updated_at.tzinfo is None
            else person.updated_at
        )

        await self._session.commit()
        return self._model_to_domain(model)

    async def delete(
        self,
        user_id: uuid.UUID,
        person_id: uuid.UUID,
    ) -> bool:
        """Delete a Person entity scoped to user_id."""
        stmt = select(PersonModel).where(
            PersonModel.id == person_id,
            PersonModel.user_id == user_id,
        )
        res = await self._session.execute(stmt)
        model = res.scalar_one_or_none()
        if not model:
            return False

        await self._session.delete(model)
        await self._session.commit()
        return True
