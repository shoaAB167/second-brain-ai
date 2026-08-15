from abc import ABC, abstractmethod
from typing import Optional
import uuid

from personal_ai.domain.experience.entity import Experience


class ExperienceRepository(ABC):
    """Abstract interface for experience persistence repositories.

    Enforces provider-independent contract for saving and retrieving raw Experience entities.
    """

    @abstractmethod
    async def create(self, experience: Experience) -> Experience:
        """Persist a new Experience domain entity.

        Args:
            experience: The Experience domain entity to create.

        Returns:
            Experience: The persisted domain entity.
        """
        pass

    @abstractmethod
    async def get_by_id(self, experience_id: uuid.UUID) -> Optional[Experience]:
        """Retrieve an Experience domain entity by ID.

        Args:
            experience_id: UUID of the experience.

        Returns:
            Optional[Experience]: The found domain entity or None.
        """
        pass
