from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from personal_ai.domain.reflection.enums import ReflectionStatus, ReflectionType


def _utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass
class Reflection:
    """Domain model representing a deterministic, evidence-grounded reflection (PR #28).

    Core Principle:
        Experiences (what happened)
            +
        Personal Patterns (behavioral hypotheses)
            ↓
        Reflection Engine
            ↓
        Reflection (observational tracking of change, reinforcement, weakening, or inconsistency)

    Safety Invariant:
        Reflections MUST remain strictly observational and uncertainty-aware.
        NEVER generates medical diagnoses, personality trait labels, or intent/motivation assumptions.
    """

    user_id: uuid.UUID
    type: ReflectionType
    observation: str
    confidence: float
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    evidence_ids: List[uuid.UUID] = field(default_factory=list)
    pattern_ids: List[uuid.UUID] = field(default_factory=list)
    time_window_days: int = 30
    domain: Optional[str] = "GENERAL"
    status: ReflectionStatus = field(default=ReflectionStatus.ACTIVE)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """Validate domain invariants, bounds, and fail closed on malformed data."""
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

        # 3. Validate type
        if isinstance(self.type, str):
            try:
                self.type = ReflectionType(self.type.upper().strip())
            except ValueError:
                raise ValueError(f"Invalid reflection type: '{self.type}'.")
        elif not isinstance(self.type, ReflectionType):
            raise ValueError(f"type must be a ReflectionType enum, got: {type(self.type).__name__}")

        # 4. Validate observation
        if not isinstance(self.observation, str) or not self.observation.strip():
            raise ValueError("Reflection observation must be a non-empty string.")
        self.observation = self.observation.strip()

        # 5. Validate confidence strictly between 0.0 and 1.0
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError(f"Confidence must be a float, got: {type(self.confidence).__name__}")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"Reflection confidence must be between 0.0 and 1.0, got: {self.confidence}.")
        self.confidence = round(float(self.confidence), 4)

        # 6. Validate evidence_ids (must be list of valid UUIDs, deduplicated)
        if not isinstance(self.evidence_ids, list):
            raise ValueError(f"evidence_ids must be a list, got: {type(self.evidence_ids).__name__}")
        normalized_ev_ids: List[uuid.UUID] = []
        seen_ev: set = set()
        for eid in self.evidence_ids:
            norm_eid: Optional[uuid.UUID] = None
            if isinstance(eid, str):
                try:
                    norm_eid = uuid.UUID(eid.strip())
                except (ValueError, AttributeError):
                    raise ValueError(f"Invalid evidence_id UUID: '{eid}'.")
            elif isinstance(eid, uuid.UUID):
                norm_eid = eid
            else:
                raise ValueError(f"evidence_ids elements must be UUIDs, got: {type(eid).__name__}")

            if norm_eid and norm_eid not in seen_ev:
                seen_ev.add(norm_eid)
                normalized_ev_ids.append(norm_eid)
        self.evidence_ids = normalized_ev_ids

        # 7. Validate pattern_ids (must be list of valid UUIDs, deduplicated)
        if not isinstance(self.pattern_ids, list):
            raise ValueError(f"pattern_ids must be a list, got: {type(self.pattern_ids).__name__}")
        normalized_pat_ids: List[uuid.UUID] = []
        seen_pat: set = set()
        for pid in self.pattern_ids:
            norm_pid: Optional[uuid.UUID] = None
            if isinstance(pid, str):
                try:
                    norm_pid = uuid.UUID(pid.strip())
                except (ValueError, AttributeError):
                    raise ValueError(f"Invalid pattern_id UUID: '{pid}'.")
            elif isinstance(pid, uuid.UUID):
                norm_pid = pid
            else:
                raise ValueError(f"pattern_ids elements must be UUIDs, got: {type(pid).__name__}")

            if norm_pid and norm_pid not in seen_pat:
                seen_pat.add(norm_pid)
                normalized_pat_ids.append(norm_pid)
        self.pattern_ids = normalized_pat_ids

        # 8. Validate time_window_days (must be integer between 1 and 90)
        if not isinstance(self.time_window_days, int) or isinstance(self.time_window_days, bool) or not (1 <= self.time_window_days <= 90):
            raise ValueError(f"time_window_days must be an integer between 1 and 90, got: {self.time_window_days}.")

        # 9. Normalize domain
        if self.domain and isinstance(self.domain, str):
            self.domain = self.domain.upper().strip()
        else:
            self.domain = "GENERAL"

        # 10. Validate status
        if isinstance(self.status, str):
            try:
                self.status = ReflectionStatus(self.status.upper().strip())
            except ValueError:
                raise ValueError(f"Invalid reflection status: '{self.status}'.")

        # 11. Ensure timezone-aware created_at
        if self.created_at and self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to clean dictionary representation."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "type": self.type.value,
            "observation": self.observation,
            "confidence": self.confidence,
            "evidence_ids": [str(eid) for eid in self.evidence_ids],
            "pattern_ids": [str(pid) for pid in self.pattern_ids],
            "time_window_days": self.time_window_days,
            "domain": self.domain,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Reflection":
        """Construct Reflection entity from dictionary representation."""
        created = (
            datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else data.get("created_at", _utc_now())
        )

        return cls(
            id=uuid.UUID(data["id"]) if isinstance(data.get("id"), str) else data.get("id", uuid.uuid4()),
            user_id=uuid.UUID(data["user_id"]) if isinstance(data["user_id"], str) else data["user_id"],
            type=ReflectionType(data["type"]),
            observation=data["observation"],
            confidence=float(data["confidence"]),
            evidence_ids=[
                uuid.UUID(eid) if isinstance(eid, str) else eid
                for eid in data.get("evidence_ids", [])
            ],
            pattern_ids=[
                uuid.UUID(pid) if isinstance(pid, str) else pid
                for pid in data.get("pattern_ids", [])
            ],
            time_window_days=int(data.get("time_window_days", 30)),
            domain=data.get("domain", "GENERAL"),
            status=ReflectionStatus(data.get("status", "ACTIVE")),
            created_at=created,
        )
