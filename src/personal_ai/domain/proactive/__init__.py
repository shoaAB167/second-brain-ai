"""Domain models and enums for Proactive Intelligence (PR #22)."""

from personal_ai.domain.proactive.enums import ProactivePriority, ProactiveSignalType
from personal_ai.domain.proactive.models import ProactiveCandidate, ProactiveSignal

__all__ = [
    "ProactiveCandidate",
    "ProactivePriority",
    "ProactiveSignal",
    "ProactiveSignalType",
]
