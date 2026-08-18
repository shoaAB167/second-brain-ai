from abc import ABC, abstractmethod
from typing import Optional
import uuid

from personal_ai.db.models import Message


class BackgroundExperienceProcessor(ABC):
    """Abstract interface for background asynchronous Experience classification and promotion."""

    @abstractmethod
    async def process_background_promotion(
        self,
        message: Message,
        user_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Process background classification and promotion for a message safely."""
        pass
