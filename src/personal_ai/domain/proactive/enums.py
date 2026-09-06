from enum import Enum


class ProactiveSignalType(str, Enum):
    """Taxonomy of initial proactive observation signal types for PR #22."""

    GOAL_INACTIVITY = "GOAL_INACTIVITY"
    COMMITMENT_MISSED = "COMMITMENT_MISSED"
    REPEATED_STATE = "REPEATED_STATE"


class ProactivePriority(str, Enum):
    """Bounded priority classification for proactive intervention candidates."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
