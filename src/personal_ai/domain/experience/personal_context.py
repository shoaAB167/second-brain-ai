from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class RetrievalDimension(str, Enum):
    """Contextual retrieval dimensions for Personal Context Retrieval (PR #18)."""

    CURRENT_STATE = "CURRENT_STATE"
    GOALS = "GOALS"
    PREFERENCES = "PREFERENCES"
    HABITS = "HABITS"
    PROJECTS = "PROJECTS"
    RELATIONSHIPS = "RELATIONSHIPS"
    EMOTIONS = "EMOTIONS"
    DECISIONS = "DECISIONS"
    PERSONALITY = "PERSONALITY"
    CONSTRAINTS = "CONSTRAINTS"
    PAST_EXPERIENCES = "PAST_EXPERIENCES"


@dataclass
class PersonalContextItem:
    """A scored and dimension-mapped memory item included in PersonalContext."""

    experience_id: uuid.UUID
    content: str
    type: Optional[str] = None
    domain: Optional[str] = None
    importance: str = "MEDIUM"
    lifecycle: str = "STABLE"
    lifecycle_status: str = "ACTIVE"
    matched_dimensions: List[RetrievalDimension] = field(default_factory=list)
    score: float = 0.0
    similarity: float = 0.0
    emotional_context: Optional[Dict[str, Any]] = None
    people_involved: Optional[List[Dict[str, Any]]] = None
    temporal_context: Optional[str] = None
    evidence_level: Optional[str] = "EXTRACTED"
    created_at: Optional[datetime] = None


@dataclass
class PersonalContext:
    """Bounded, dimension-aware personal context assembled for an authenticated user query."""

    user_id: uuid.UUID
    query: str
    detected_dimensions: List[RetrievalDimension] = field(default_factory=list)
    items: List[PersonalContextItem] = field(default_factory=list)
    total_candidates: int = 0

    @property
    def is_empty(self) -> bool:
        """Return True if no context items were retrieved."""
        return len(self.items) == 0
