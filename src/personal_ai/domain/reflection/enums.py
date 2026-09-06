from enum import Enum


class ReflectionType(str, Enum):
    """Controlled taxonomy for evidence-grounded behavioral reflections (PR #28)."""

    PATTERN_REINFORCEMENT = "PATTERN_REINFORCEMENT"
    PATTERN_WEAKENING = "PATTERN_WEAKENING"
    PATTERN_CHANGE = "PATTERN_CHANGE"
    PATTERN_INCONSISTENCY = "PATTERN_INCONSISTENCY"


class ReflectionStatus(str, Enum):
    """Lifecycle status states for generated reflections."""

    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
