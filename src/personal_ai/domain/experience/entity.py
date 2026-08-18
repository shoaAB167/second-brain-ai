from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid

from personal_ai.domain.experience.enums import ExperienceSource, ExperienceStatus


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass
class Experience:
    """Domain entity representing a raw, uninterpreted life experience observation.

    Pure Python domain entity. Does not depend on FastAPI, SQLAlchemy, or Pydantic.
    Original content is treated as raw source-of-truth and is never transformed or summarized.
    """

    content: str
    source: ExperienceSource
    user_id: Optional[str] = None
    source_message_id: Optional[uuid.UUID] = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    status: ExperienceStatus = field(default=ExperienceStatus.RECEIVED)
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
