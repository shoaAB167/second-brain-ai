from enum import Enum


class PatternStatus(str, Enum):
    """Lifecycle status states for personal behavioral patterns."""

    HYPOTHESIS = "HYPOTHESIS"
    CONFIRMED = "CONFIRMED"
    WEAKENED = "WEAKENED"
    SUPERSEDED = "SUPERSEDED"


class PatternDomain(str, Enum):
    """Domain categorization for personal behavioral patterns."""

    CAREER = "CAREER"
    FITNESS = "FITNESS"
    RELATIONSHIPS = "RELATIONSHIPS"
    HEALTH = "HEALTH"
    PROJECTS = "PROJECTS"
    LEARNING = "LEARNING"
    FINANCE = "FINANCE"
    SOCIAL = "SOCIAL"
    GENERAL = "GENERAL"
