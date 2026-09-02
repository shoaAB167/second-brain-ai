from abc import ABC, abstractmethod
from typing import List, Optional
import uuid

from personal_ai.domain.experience.enums import ExperienceRelationshipType
from personal_ai.domain.experience.relationship import ExperienceRelationship


class ExperienceRelationshipRepository(ABC):
    """Abstract interface for experience relationship persistence repositories."""

    @abstractmethod
    async def create(self, relationship: ExperienceRelationship) -> ExperienceRelationship:
        """Persist a new ExperienceRelationship domain entity.

        Args:
            relationship: Domain relationship entity to persist.

        Returns:
            ExperienceRelationship: Persisted domain entity.
        """
        pass

    @abstractmethod
    async def get_by_source_id(self, source_id: uuid.UUID) -> List[ExperienceRelationship]:
        """Retrieve all relationships where the given experience is the source.

        Args:
            source_id: UUID of the source experience.

        Returns:
            List[ExperienceRelationship]: List of matching relationships.
        """
        pass

    @abstractmethod
    async def get_by_target_id(self, target_id: uuid.UUID) -> List[ExperienceRelationship]:
        """Retrieve all relationships where the given experience is the target.

        Args:
            target_id: UUID of the target experience.

        Returns:
            List[ExperienceRelationship]: List of matching relationships.
        """
        pass

    @abstractmethod
    async def get_by_experience_id(self, experience_id: uuid.UUID) -> List[ExperienceRelationship]:
        """Retrieve all relationships where the given experience is either source or target.

        Args:
            experience_id: UUID of the experience.

        Returns:
            List[ExperienceRelationship]: List of matching relationships.
        """
        pass

    @abstractmethod
    async def exists(
        self,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
        relationship_type: ExperienceRelationshipType,
    ) -> bool:
        """Check if a specific relationship already exists between source and target.

        Args:
            source_id: UUID of source experience.
            target_id: UUID of target experience.
            relationship_type: The relationship type.

        Returns:
            bool: True if already recorded, False otherwise.
        """
        pass
