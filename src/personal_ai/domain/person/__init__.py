"""Domain models and interfaces for Person / Relationship layer (PR #29)."""

from personal_ai.domain.person.entity import Person, PersonalPersonContext
from personal_ai.domain.person.enums import RelationshipType
from personal_ai.domain.person.repository import PersonRepository

__all__ = [
    "Person",
    "PersonalPersonContext",
    "RelationshipType",
    "PersonRepository",
]
