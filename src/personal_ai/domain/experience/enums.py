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
    """Lifecycle status states for experiences."""
    RECEIVED = "RECEIVED"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class ExperienceType(str, Enum):
    """Taxonomy types for classified experiences."""
    GOAL = "GOAL"
    DECISION = "DECISION"
    PREFERENCE = "PREFERENCE"
    FACT = "FACT"
    EVENT = "EVENT"
    RELATIONSHIP = "RELATIONSHIP"
    EMOTION_STATE = "EMOTION_STATE"
    HABIT = "HABIT"
    PROJECT = "PROJECT"
    OTHER = "OTHER"
