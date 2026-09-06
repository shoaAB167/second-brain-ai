from enum import Enum


class RelationshipType(str, Enum):
    """Controlled, generic taxonomy for interpersonal relationship types (PR #29).

    Safety Invariant:
        Keeps relationship classifications minimal, standard, and descriptive.
        Does NOT infer attachment styles, relationship health, or psychological traits.
    """

    FAMILY = "FAMILY"
    FRIEND = "FRIEND"
    PARTNER = "PARTNER"
    COLLEAGUE = "COLLEAGUE"
    ACQUAINTANCE = "ACQUAINTANCE"
    MENTOR = "MENTOR"
    OTHER = "OTHER"
