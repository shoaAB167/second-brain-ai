from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid

from personal_ai.domain.experience.enums import ExperienceRelationshipType


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass
class ExperienceRelationship:
    """Domain entity representing a directional relationship between two experiences.

    Pure Python domain entity.
    `source_experience_id` is typically the newer/incoming experience.
    `target_experience_id` is the existing candidate experience it relates to.
    """

    source_experience_id: uuid.UUID
    target_experience_id: uuid.UUID
    relationship_type: ExperienceRelationshipType
    confidence: float
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    reason: Optional[str] = None
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate domain invariants for experience relationships."""
        if isinstance(self.relationship_type, str):
            try:
                self.relationship_type = ExperienceRelationshipType(self.relationship_type.upper().strip())
            except ValueError:
                raise ValueError(f"Invalid experience relationship type: '{self.relationship_type}'.")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Relationship confidence must be between 0.0 and 1.0, got {self.confidence}.")

        if self.source_experience_id == self.target_experience_id:
            raise ValueError("An experience cannot have a relationship with itself.")
