from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
import uuid

from personal_ai.domain.experience.enums import ExperienceSource, ExperienceStatus, ExperienceType


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass
class Experience:
    """Domain entity representing a structured or raw life experience observation.

    Pure Python domain entity. Does not depend on FastAPI, SQLAlchemy, or Pydantic.
    Original provenance is preserved via source_message_id and user_id.
    """

    content: str
    source: ExperienceSource
    user_id: Optional[str] = None
    source_message_id: Optional[uuid.UUID] = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    type: Optional[ExperienceType] = None
    domain: Optional[str] = None
    status: ExperienceStatus = field(default=ExperienceStatus.RECEIVED)
    extraction_confidence: Optional[float] = None
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    embedding_status: str = "PENDING"  # PENDING, COMPLETED, FAILED
    embedded_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate domain business rules upon instantiation."""
        if not isinstance(self.content, str):
            raise ValueError("Experience content must be a string.")

        if not self.content or not self.content.strip():
            raise ValueError("Experience content cannot be empty or whitespace-only.")

        if isinstance(self.source, str):
            try:
                self.source = ExperienceSource(self.source)
            except ValueError:
                raise ValueError(f"Invalid experience source: '{self.source}'.")

        if isinstance(self.status, str):
            try:
                self.status = ExperienceStatus(self.status)
            except ValueError:
                raise ValueError(f"Invalid experience status: '{self.status}'.")

        if isinstance(self.type, str) and self.type:
            val_str = self.type.upper().strip()
            if val_str == "EMOTION":
                val_str = "EMOTION_STATE"
            try:
                self.type = ExperienceType(val_str)
            except ValueError:
                raise ValueError(f"Invalid experience type: '{self.type}'.")
