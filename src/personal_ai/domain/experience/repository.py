from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import uuid

from personal_ai.domain.experience.entity import Experience


class ExperienceRepository(ABC):
    """Abstract interface for experience persistence repositories.

    Enforces provider-independent contract for saving, updating, and retrieving Experience entities.
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
    async def update(self, experience: Experience) -> Experience:
        """Update an existing Experience domain entity.

        Args:
            experience: The Experience domain entity to update.

        Returns:
            Experience: The updated domain entity.
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

    @abstractmethod
    async def get_by_source_message_id(self, source_message_id: uuid.UUID) -> Optional[Experience]:
        """Retrieve an Experience domain entity by source message ID provenance.

        Args:
            source_message_id: UUID of the originating conversation message.

        Returns:
            Optional[Experience]: Found domain entity or None.
        """
        pass

    @abstractmethod
    async def search_by_vector(
        self,
        user_id: uuid.UUID,
        query_vector: List[float],
        limit: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Tuple[Experience, float]]:
        """Search experiences for a specific user ordered by semantic similarity to query_vector.

        Strictly user-scoped at the database layer. Ignores experiences without valid completed embeddings.

        Args:
            user_id: Target user UUID (strict user scoping).
            query_vector: Embedding vector of the search query.
            limit: Maximum number of results to return (default: 5).
            threshold: Optional minimum cosine similarity score threshold (in [-1.0, 1.0]).

        Returns:
            List[Tuple[Experience, float]]: List of (Experience, similarity_score) tuples ordered by descending similarity.
        """
        pass
