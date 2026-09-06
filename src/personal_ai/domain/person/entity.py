from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from personal_ai.domain.experience.entity import Experience
from personal_ai.domain.person.enums import RelationshipType


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


def _normalize_relationship_type(val: Any) -> RelationshipType:
    """Normalize any relationship type representation to RelationshipType enum. Defaults to OTHER."""
    if val is None:
        return RelationshipType.OTHER
    if isinstance(val, RelationshipType):
        return val
    if isinstance(val, str):
        val_str = val.upper().strip()
        try:
            return RelationshipType(val_str)
        except ValueError:
            return RelationshipType.OTHER
    return RelationshipType.OTHER


@dataclass
class Person:
    """Domain model representing a stable individual in the user's life (PR #29).

    Core Principle:
        Person = stable entity representing someone in the user's life.
        Experience = event/memory involving that person.

    Safety Invariant:
        Reflections and People records MUST remain strictly observational.
        NEVER stores relationship health scores, trust scores, attachment styles,
        or psychological/personality conclusions.
    """

    user_id: uuid.UUID
    name: str
    relationship_type: RelationshipType = field(default=RelationshipType.OTHER)
    notes: Optional[str] = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """Validate domain invariants, bounds, and fail closed on invalid data."""
        # 1. Validate user_id
        if isinstance(self.user_id, str):
            try:
                self.user_id = uuid.UUID(self.user_id.strip())
            except (ValueError, AttributeError):
                raise ValueError(f"Invalid user_id UUID: '{self.user_id}'.")
        elif not isinstance(self.user_id, uuid.UUID):
            raise ValueError(f"user_id must be a UUID, got: {type(self.user_id).__name__}")

        # 2. Validate id
        if isinstance(self.id, str):
            try:
                self.id = uuid.UUID(self.id.strip())
            except (ValueError, AttributeError):
                raise ValueError(f"Invalid id UUID: '{self.id}'.")
        elif not isinstance(self.id, uuid.UUID):
            raise ValueError(f"id must be a UUID, got: {type(self.id).__name__}")

        # 3. Validate name
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Person name must be a non-empty string.")
        self.name = self.name.strip()

        # 4. Validate and normalize relationship_type
        self.relationship_type = _normalize_relationship_type(self.relationship_type)

        # 5. Normalize notes
        if self.notes is not None:
            self.notes = str(self.notes).strip() or None

        # 6. Ensure timezone-aware datetimes
        if self.created_at and self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        if self.updated_at and self.updated_at.tzinfo is None:
            self.updated_at = self.updated_at.replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Person to dictionary representation."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "name": self.name,
            "relationship_type": self.relationship_type.value,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Person":
        """Construct Person entity from dictionary representation with fail-closed normalization."""
        if not data or not isinstance(data, dict):
            raise ValueError("Person data must be a dictionary.")

        raw_user_id = data.get("user_id")
        if raw_user_id is None:
            raise ValueError("user_id is required in Person dictionary.")
        if isinstance(raw_user_id, str):
            try:
                user_id_val = uuid.UUID(raw_user_id.strip())
            except (ValueError, AttributeError):
                raise ValueError(f"Invalid user_id UUID: '{raw_user_id}'.")
        elif isinstance(raw_user_id, uuid.UUID):
            user_id_val = raw_user_id
        else:
            raise ValueError(f"user_id must be a UUID, got: {type(raw_user_id).__name__}")

        raw_id = data.get("id")
        id_val = uuid.uuid4()
        if raw_id is not None:
            if isinstance(raw_id, str):
                try:
                    id_val = uuid.UUID(raw_id.strip())
                except (ValueError, AttributeError):
                    raise ValueError(f"Invalid id UUID: '{raw_id}'.")
            elif isinstance(raw_id, uuid.UUID):
                id_val = raw_id
            else:
                raise ValueError(f"id must be a UUID, got: {type(raw_id).__name__}")

        name_val = data.get("name")
        if not isinstance(name_val, str) or not name_val.strip():
            raise ValueError("Person name must be a non-empty string.")

        rel_type = _normalize_relationship_type(data.get("relationship_type"))

        created = (
            datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else data.get("created_at", _utc_now())
        )
        updated = (
            datetime.fromisoformat(data["updated_at"])
            if isinstance(data.get("updated_at"), str)
            else data.get("updated_at", _utc_now())
        )

        return cls(
            id=id_val,
            user_id=user_id_val,
            name=name_val.strip(),
            relationship_type=rel_type,
            notes=data.get("notes"),
            created_at=created,
            updated_at=updated,
        )


@dataclass
class PersonalPersonContext:
    """Bounded, person-scoped context container (PR #29).

    Provides structured access to experiences involving a specific person without
    speculative psychological scoring or graph complexity.
    """

    person: Person
    relevant_experiences: List[Experience] = field(default_factory=list)
    total_experiences: int = 0
