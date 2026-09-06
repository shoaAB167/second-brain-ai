from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import uuid


@dataclass
class EmotionalContext:
    """Domain model representing human emotional context attached to an experience.

    All fields are optional. Bounded intensity must be between 0.0 and 1.0 if specified.
    """

    emotion: Optional[str] = None
    intensity: Optional[float] = None
    trigger: Optional[str] = None
    need: Optional[str] = None
    impact: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate bounded intensity and types."""
        if self.emotion is not None:
            if not isinstance(self.emotion, str) or not self.emotion.strip():
                self.emotion = None
            else:
                self.emotion = self.emotion.strip().lower()

        if self.intensity is not None:
            try:
                val = float(self.intensity)
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"Emotional intensity must be between 0.0 and 1.0, got: {self.intensity}")
                self.intensity = round(val, 2)
            except (TypeError, ValueError) as exc:
                if isinstance(exc, ValueError) and "between 0.0 and 1.0" in str(exc):
                    raise
                raise ValueError(f"Invalid emotional intensity: '{self.intensity}'")

        if self.trigger is not None:
            self.trigger = str(self.trigger).strip() or None

        if self.need is not None:
            self.need = str(self.need).strip() or None

        if self.impact is not None:
            self.impact = str(self.impact).strip() or None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to clean dictionary representation."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["EmotionalContext"]:
        """Construct from raw dictionary."""
        if not data or not isinstance(data, dict):
            return None
        return cls(
            emotion=data.get("emotion"),
            intensity=data.get("intensity"),
            trigger=data.get("trigger"),
            need=data.get("need"),
            impact=data.get("impact"),
        )


@dataclass
class PersonInvolved:
    """Domain model representing contextual individuals associated with an experience (PR #12 & PR #29)."""

    name: str
    role: Optional[str] = None
    person_id: Optional[uuid.UUID] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Person name must be a non-empty string.")
        self.name = self.name.strip()
        if self.role is not None:
            self.role = str(self.role).strip() or None
        if isinstance(self.person_id, str):
            cleaned = self.person_id.strip()
            if cleaned:
                try:
                    self.person_id = uuid.UUID(cleaned)
                except (ValueError, AttributeError):
                    self.person_id = cleaned
            else:
                self.person_id = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to clean dictionary representation."""
        res: Dict[str, Any] = {"name": self.name}
        if self.role is not None:
            res["role"] = self.role
        if self.person_id is not None:
            res["person_id"] = str(self.person_id)
        return res

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["PersonInvolved"]:
        """Construct from raw dictionary."""
        if not data or not isinstance(data, dict) or not data.get("name"):
            return None
        pid: Any = None
        if "person_id" in data and data["person_id"] is not None:
            raw_val = data["person_id"]
            if isinstance(raw_val, uuid.UUID):
                pid = raw_val
            elif isinstance(raw_val, str) and raw_val.strip():
                try:
                    pid = uuid.UUID(raw_val.strip())
                except (ValueError, AttributeError):
                    pid = raw_val.strip()
            else:
                pid = raw_val
        return cls(
            name=data["name"],
            role=data.get("role"),
            person_id=pid,
        )
