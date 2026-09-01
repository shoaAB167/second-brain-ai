from enum import Enum


class ExperienceSource(str, Enum):
    """Controlled source representations for raw experiences."""
    CHAT = "CHAT"
    FILE = "FILE"
    EMAIL = "EMAIL"
    CALENDAR = "CALENDAR"
    GITHUB = "GITHUB"
    VOICE = "VOICE"
    API = "API"
    MANUAL = "MANUAL"
    TOOL = "TOOL"


class ExperienceStatus(str, Enum):
    """Processing status states for experiences."""
    RECEIVED = "RECEIVED"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class ExperienceType(str, Enum):
    """Taxonomy types for classified experiences."""
    FACT = "FACT"
    GOAL = "GOAL"
    PREFERENCE = "PREFERENCE"
    HABIT = "HABIT"
    PROJECT = "PROJECT"
    EVENT = "EVENT"
    STATE = "STATE"
    DECISION = "DECISION"
    RELATIONSHIP = "RELATIONSHIP"
    EMOTION_STATE = "EMOTION_STATE"
    OTHER = "OTHER"


class ExperienceImportance(str, Enum):
    """Bounded importance levels for long-term personal memory."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ExperienceLifecycle(str, Enum):
    """Temporal durability and lifecycle scopes for personal memories."""
    STABLE = "STABLE"
    RECURRING = "RECURRING"
    TEMPORARY = "TEMPORARY"
    TIME_BOUND = "TIME_BOUND"
