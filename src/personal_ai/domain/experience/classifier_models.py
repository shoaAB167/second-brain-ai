from typing import Optional
from pydantic import BaseModel, Field, field_validator

from personal_ai.domain.experience.enums import ExperienceType


class ClassificationResult(BaseModel):
    """Structured result produced by the AI Experience Classifier.

    Represents probabilistic classification metrics for user messages.
    Does NOT directly create or modify Experience records in the database.
    """

    is_experience: bool = Field(
        ...,
        description="True if message contains meaningful persistent life/personal information.",
    )
    type: Optional[ExperienceType] = Field(
        default=None,
        description="Broad taxonomy category of the experience, if is_experience is True.",
    )
    importance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Importance score between 0.0 (trivial) and 1.0 (vital).",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classifier confidence score between 0.0 (uncertain) and 1.0 (certain).",
    )
    raw_model: Optional[str] = Field(
        default=None,
        description="Identifier of the LLM model that generated the classification.",
    )

    @field_validator("importance", "confidence")
    @classmethod
    def validate_score_range(cls, value: float) -> float:
        """Validate that score is strictly bounded within [0.0, 1.0]."""
        if not isinstance(value, (int, float)):
            raise ValueError("Score must be a numeric float.")
        val_float = float(value)
        if val_float < 0.0 or val_float > 1.0:
            raise ValueError(f"Score '{value}' must be bounded between 0.0 and 1.0 inclusive.")
        return val_float

    @field_validator("type", mode="before")
    @classmethod
    def validate_type_enum(cls, value: Optional[str]) -> Optional[ExperienceType]:
        """Convert string to ExperienceType enum if present."""
        if value is None or value == "":
            return None
        if isinstance(value, ExperienceType):
            return value
        try:
            return ExperienceType(str(value).upper())
        except ValueError:
            return ExperienceType.OTHER
