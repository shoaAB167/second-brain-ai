from abc import ABC, abstractmethod
from typing import List, Optional
import uuid

from personal_ai.domain.pattern.entity import PersonalPattern
from personal_ai.domain.pattern.enums import PatternStatus


class PersonalPatternRepository(ABC):
    """Abstract repository interface for PersonalPattern persistence."""

    @abstractmethod
    async def create(self, pattern: PersonalPattern) -> PersonalPattern:
        """Persist a new PersonalPattern entity.

        Args:
            pattern: The domain pattern entity to persist.

        Returns:
            PersonalPattern: The persisted pattern entity.
        """
        pass

    async def save(self, pattern: PersonalPattern) -> PersonalPattern:
        """Alias for create/persist."""
        return await self.create(pattern)

    @abstractmethod
    async def update(self, pattern: PersonalPattern) -> PersonalPattern:
        """Update an existing PersonalPattern entity.

        Args:
            pattern: The domain pattern entity with updated fields.

        Returns:
            PersonalPattern: The updated pattern entity.
        """
        pass

    @abstractmethod
    async def get_by_id(
        self,
        pattern_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
    ) -> Optional[PersonalPattern]:
        """Retrieve a PersonalPattern by its unique ID, optionally scoped to user_id.

        Args:
            pattern_id: Pattern UUID.
            user_id: Optional user UUID for strict isolation.

        Returns:
            Optional[PersonalPattern]: The found pattern entity or None.
        """
        pass

    @abstractmethod
    async def list_by_user(
        self,
        user_id: uuid.UUID,
        status: Optional[PatternStatus] = None,
        domain: Optional[str] = None,
    ) -> List[PersonalPattern]:
        """List patterns belonging to a specific authenticated user.

        Args:
            user_id: Authenticated user UUID for strict isolation.
            status: Optional filter on PatternStatus.
            domain: Optional filter on domain name.

        Returns:
            List[PersonalPattern]: List of matching patterns.
        """
        pass

    @abstractmethod
    async def get_active_patterns(
        self,
        user_id: uuid.UUID,
        domain: Optional[str] = None,
    ) -> List[PersonalPattern]:
        """Retrieve active (non-superseded) pattern hypotheses and confirmed patterns for a user.

        Args:
            user_id: Authenticated user UUID.
            domain: Optional domain filter.

        Returns:
            List[PersonalPattern]: List of active patterns (HYPOTHESIS or CONFIRMED).
        """
        pass

