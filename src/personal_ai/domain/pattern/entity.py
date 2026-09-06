from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import uuid

from personal_ai.domain.pattern.enums import PatternDomain, PatternStatus


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass
class PersonalPattern:
    """Domain model representing a personal behavioral pattern hypothesis or confirmed pattern.

    Core Principle:
        Observation -> Repeated Evidence -> Pattern Hypothesis -> Confidence -> Personal Model

    Safety Invariant:
        Describes observable behavior with uncertainty language ('appears to', 'tends to').
        NEVER diagnoses medical/psychological conditions or asserts permanent personality traits.
    """

    user_id: uuid.UUID
    description: str
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    domain: str = "GENERAL"
    evidence_ids: List[uuid.UUID] = field(default_factory=list)
    confidence: float = 0.55
    status: PatternStatus = field(default=PatternStatus.HYPOTHESIS)
    first_observed_at: datetime = field(default_factory=utc_now)
    last_observed_at: datetime = field(default_factory=utc_now)
    superseded_by_id: Optional[uuid.UUID] = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate domain invariants and bounds."""
        # Validate user_id
        if isinstance(self.user_id, str):
            try:
                self.user_id = uuid.UUID(self.user_id)
            except ValueError:
                raise ValueError(f"Invalid user_id UUID: '{self.user_id}'.")
        elif not isinstance(self.user_id, uuid.UUID):
            raise ValueError(f"user_id must be a UUID, got: {type(self.user_id).__name__}")

        # Validate description
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("Pattern description must be a non-empty string.")
        self.description = self.description.strip()

        # Validate confidence strictly between 0.0 and 1.0
        if not isinstance(self.confidence, (int, float)):
            raise ValueError(f"Confidence must be a float, got: {type(self.confidence).__name__}")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"Pattern confidence must be between 0.0 and 1.0, got {self.confidence}.")
        self.confidence = round(float(self.confidence), 4)

        # Validate status
        if isinstance(self.status, str):
            try:
                self.status = PatternStatus(self.status.upper().strip())
            except ValueError:
                raise ValueError(f"Invalid pattern status: '{self.status}'.")

        # Normalize domain
        if isinstance(self.domain, PatternDomain):
            self.domain = self.domain.value
        elif isinstance(self.domain, str):
            self.domain = self.domain.upper().strip() or "GENERAL"

        # Validate evidence_ids
        normalized_evidence: List[uuid.UUID] = []
        for eid in self.evidence_ids:
            if isinstance(eid, str):
                try:
                    normalized_evidence.append(uuid.UUID(eid))
                except ValueError:
                    continue
            elif isinstance(eid, uuid.UUID):
                normalized_evidence.append(eid)
        self.evidence_ids = normalized_evidence

        # Ensure timezone-aware datetimes
        if self.first_observed_at and self.first_observed_at.tzinfo is None:
            self.first_observed_at = self.first_observed_at.replace(tzinfo=timezone.utc)
        if self.last_observed_at and self.last_observed_at.tzinfo is None:
            self.last_observed_at = self.last_observed_at.replace(tzinfo=timezone.utc)
        if self.created_at and self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        if self.updated_at and self.updated_at.tzinfo is None:
            self.updated_at = self.updated_at.replace(tzinfo=timezone.utc)

    def add_evidence(self, experience_id: uuid.UUID, observed_at: Optional[datetime] = None) -> None:
        """Attach new supporting experience evidence and update temporal tracking."""
        if experience_id not in self.evidence_ids:
            self.evidence_ids.append(experience_id)

        obs_time = observed_at or utc_now()
        if obs_time.tzinfo is None:
            obs_time = obs_time.replace(tzinfo=timezone.utc)

        if obs_time > self.last_observed_at:
            self.last_observed_at = obs_time
        if obs_time < self.first_observed_at:
            self.first_observed_at = obs_time

        self.updated_at = utc_now()

    def update_confidence(self, new_confidence: float) -> None:
        """Update confidence bounded within [0.0, 1.0]."""
        if not (0.0 <= new_confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got: {new_confidence}")
        self.confidence = round(new_confidence, 4)
        self.updated_at = utc_now()

    def weaken(self, confidence_reduction: float = 0.15) -> None:
        """Weaken pattern confidence when contradictory/inconsistent evidence is observed."""
        new_conf = max(0.10, self.confidence - confidence_reduction)
        self.confidence = round(new_conf, 4)
        if self.confidence < 0.40:
            self.status = PatternStatus.WEAKENED
        self.updated_at = utc_now()

    def supersede_with(self, new_pattern_id: Union[uuid.UUID, str]) -> None:
        """Mark pattern as superseded by a newer, refined pattern without deleting historical record."""
        if not new_pattern_id:
            raise ValueError("new_pattern_id is required when superseding a pattern.")
        if isinstance(new_pattern_id, str):
            try:
                new_pattern_id = uuid.UUID(new_pattern_id)
            except ValueError:
                raise ValueError(f"Invalid new_pattern_id UUID: '{new_pattern_id}'.")
        if not isinstance(new_pattern_id, uuid.UUID):
            raise ValueError(f"new_pattern_id must be a UUID, got {type(new_pattern_id).__name__}")

        if new_pattern_id == self.id:
            raise ValueError("A pattern cannot supersede itself.")

        self.status = PatternStatus.SUPERSEDED
        self.superseded_by_id = new_pattern_id
        self.updated_at = utc_now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to clean dictionary representation."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "description": self.description,
            "domain": self.domain,
            "evidence_ids": [str(eid) for eid in self.evidence_ids],
            "confidence": self.confidence,
            "status": self.status.value,
            "first_observed_at": self.first_observed_at.isoformat(),
            "last_observed_at": self.last_observed_at.isoformat(),
            "superseded_by_id": str(self.superseded_by_id) if self.superseded_by_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalPattern":
        """Construct from dictionary representation."""
        first_obs = (
            datetime.fromisoformat(data["first_observed_at"])
            if isinstance(data.get("first_observed_at"), str)
            else data["first_observed_at"]
        )
        last_obs = (
            datetime.fromisoformat(data["last_observed_at"])
            if isinstance(data.get("last_observed_at"), str)
            else data["last_observed_at"]
        )
        created = (
            datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else data.get("created_at", utc_now())
        )
        updated = (
            datetime.fromisoformat(data["updated_at"])
            if isinstance(data.get("updated_at"), str)
            else data.get("updated_at", utc_now())
        )

        return cls(
            id=uuid.UUID(data["id"]) if isinstance(data.get("id"), str) else data.get("id", uuid.uuid4()),
            user_id=uuid.UUID(data["user_id"]) if isinstance(data["user_id"], str) else data["user_id"],
            description=data["description"],
            domain=data.get("domain", "GENERAL"),
            evidence_ids=[
                uuid.UUID(eid) if isinstance(eid, str) else eid
                for eid in data.get("evidence_ids", [])
            ],
            confidence=float(data.get("confidence", 0.55)),
            status=PatternStatus(data.get("status", "HYPOTHESIS")),
            first_observed_at=first_obs,
            last_observed_at=last_obs,
            superseded_by_id=(
                uuid.UUID(data["superseded_by_id"])
                if data.get("superseded_by_id")
                else None
            ),
            created_at=created,
            updated_at=updated,
        )
