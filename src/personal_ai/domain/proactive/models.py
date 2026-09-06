from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import uuid

from personal_ai.domain.proactive.enums import ProactivePriority, ProactiveSignalType


@dataclass(frozen=True)
class ProactiveSignal:
    """Domain model representing a detected observation signal.

    Captures an observed situation with confidence and provenance back to source experiences.
    """

    type: ProactiveSignalType
    reason: str
    confidence: float
    related_experience_ids: List[uuid.UUID] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate confidence boundaries and enum types."""
        if not isinstance(self.confidence, (int, float)):
            raise ValueError(f"Confidence must be a numeric float, got: {type(self.confidence).__name__}")

        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got: {self.confidence}")

        object.__setattr__(self, "confidence", round(float(self.confidence), 4))

        if not isinstance(self.type, ProactiveSignalType):
            try:
                object.__setattr__(self, "type", ProactiveSignalType(self.type))
            except ValueError:
                raise ValueError(f"Invalid signal type: '{self.type}'.")

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Signal reason must be a non-empty string.")

        if not isinstance(self.related_experience_ids, list):
            raise ValueError("related_experience_ids must be a list of UUIDs.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to clean dictionary representation."""
        return {
            "type": self.type.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "related_experience_ids": [str(eid) for eid in self.related_experience_ids],
        }


@dataclass(frozen=True)
class ProactiveCandidate:
    """Domain model representing a proposed intervention candidate for future consideration.

    This is a passive proposal data container, NOT an action or instruction.
    It does not execute tools, send notifications, or modify memory.
    """

    signal_type: ProactiveSignalType
    reason: str
    confidence: float
    priority: ProactivePriority
    suggested_action: str
    related_experience_ids: List[uuid.UUID] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate confidence, priority, and enum types."""
        if not isinstance(self.confidence, (int, float)):
            raise ValueError(f"Confidence must be a numeric float, got: {type(self.confidence).__name__}")

        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got: {self.confidence}")

        object.__setattr__(self, "confidence", round(float(self.confidence), 4))

        if not isinstance(self.signal_type, ProactiveSignalType):
            try:
                object.__setattr__(self, "signal_type", ProactiveSignalType(self.signal_type))
            except ValueError:
                raise ValueError(f"Invalid signal type: '{self.signal_type}'.")

        if not isinstance(self.priority, ProactivePriority):
            try:
                object.__setattr__(self, "priority", ProactivePriority(self.priority))
            except ValueError:
                raise ValueError(f"Invalid priority: '{self.priority}'.")

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Candidate reason must be a non-empty string.")

        if not isinstance(self.suggested_action, str) or not self.suggested_action.strip():
            raise ValueError("Candidate suggested_action must be a non-empty string.")

        if not isinstance(self.related_experience_ids, list):
            raise ValueError("related_experience_ids must be a list of UUIDs.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to clean dictionary representation."""
        return {
            "signal_type": self.signal_type.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "priority": self.priority.value,
            "suggested_action": self.suggested_action,
            "related_experience_ids": [str(eid) for eid in self.related_experience_ids],
        }
