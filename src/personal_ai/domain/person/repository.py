from abc import ABC, abstractmethod
from typing import List, Optional
import uuid

from personal_ai.domain.person.entity import Person


class PersonRepository(ABC):
    """Abstract repository interface for Person persistence (PR #29).

    All repository operations are strictly scoped to the authenticated user_id.
    """

    @abstractmethod
    async def create(self, person: Person) -> Person:
        """Persist a new Person entity.

        Args:
            person: The domain Person entity to persist.

        Returns:
            Person: The persisted Person entity.
        """
        pass

    async def save(self, person: Person) -> Person:
        """Alias for create/persist."""
        return await self.create(person)

    @abstractmethod
    async def get_by_id(
        self,
        user_id: uuid.UUID,
        person_id: uuid.UUID,
    ) -> Optional[Person]:
        """Retrieve a Person by unique ID scoped to user_id.

        Args:
            user_id: Authenticated user UUID for strict isolation.
            person_id: Person unique UUID.

        Returns:
            Optional[Person]: The found Person entity or None if not found or cross-user.
        """
        pass

    @abstractmethod
    async def list(
        self,
        user_id: uuid.UUID,
    ) -> List[Person]:
        """List all Person records belonging to a specific authenticated user.

        Args:
            user_id: Authenticated user UUID for strict isolation.

        Returns:
            List[Person]: List of Person entities.
        """
        pass

    @abstractmethod
    async def find_by_name(
        self,
        user_id: uuid.UUID,
        name: str,
    ) -> Optional[Person]:
        """Find a Person by case-insensitive exact name match for an authenticated user.

        Args:
            user_id: Authenticated user UUID for strict isolation.
            name: Exact name of the person (matched case-insensitively).

        Returns:
            Optional[Person]: The matching Person entity or None.
        """
        pass

    @abstractmethod
    async def update(
        self,
        user_id: uuid.UUID,
        person: Person,
    ) -> Person:
        """Update an existing Person entity scoped to user_id.

        Args:
            user_id: Authenticated user UUID.
            person: The updated Person entity.

        Returns:
            Person: The updated Person entity.
        """
        pass

    @abstractmethod
    async def delete(
        self,
        user_id: uuid.UUID,
        person_id: uuid.UUID,
    ) -> bool:
        """Delete a Person entity scoped to user_id.

        Args:
            user_id: Authenticated user UUID.
            person_id: Person UUID to delete.

        Returns:
            bool: True if deleted, False if not found or cross-user.
        """
        pass
