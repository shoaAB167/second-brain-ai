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
