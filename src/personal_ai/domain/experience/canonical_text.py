from personal_ai.domain.experience.entity import Experience


def build_experience_embedding_text(experience: Experience) -> str:
    """Construct deterministic canonical embedding input text from structured Experience attributes.

    Excludes raw conversation history, system metadata, database IDs, and sensitive credentials.

    Args:
        experience: Validated Experience domain entity.

    Returns:
        str: Deterministic text representation formatted for vector embedding generation.
    """
    parts = []

    type_val = (
        experience.type.value
        if hasattr(experience.type, "value")
        else (str(experience.type) if experience.type else "UNSPECIFIED")
    )
    parts.append(f"Type: {type_val}")

    if experience.domain and experience.domain.strip():
        parts.append(f"Domain: {experience.domain.strip()}")

    status_val = (
        experience.status.value
        if hasattr(experience.status, "value")
        else str(experience.status)
    )
    parts.append(f"Status: {status_val}")

    parts.append(f"Content: {experience.content.strip()}")

    return " | ".join(parts)
