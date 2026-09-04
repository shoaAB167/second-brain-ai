from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


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
    """Domain model representing contextual individuals associated with an experience."""

    name: str
    role: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Person name must be a non-empty string.")
        self.name = self.name.strip()
        if self.role is not None:
            self.role = str(self.role).strip() or None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to clean dictionary representation."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["PersonInvolved"]:
        """Construct from raw dictionary."""
        if not data or not isinstance(data, dict) or not data.get("name"):
            return None
        return cls(
            name=data["name"],
            role=data.get("role"),
        )
