from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from personal_ai.domain.experience.enums import ExperienceType


class ClassificationResult(BaseModel):
    """Structured result produced by the AI Experience Classifier.

    Represents probabilistic classification metrics for user messages.
    Fails closed on invalid experience types or inconsistent fields.
    Does NOT directly create or modify Experience records in the database.
    """

    is_experience: bool = Field(
        ...,
        description="True if message contains meaningful persistent life/personal information.",
    )
    type: Optional[ExperienceType] = Field(
        default=None,
        description="Broad taxonomy category of the experience, required if is_experience is True.",
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
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional short explanation of classifier reasoning.",
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
        """Convert string to ExperienceType enum. Fails closed on invalid strings."""
        if value is None or value == "" or str(value).lower() in ("null", "none"):
            return None
        if isinstance(value, ExperienceType):
            return value
        val_str = str(value).upper().strip()
        if val_str == "EMOTION":
            val_str = "EMOTION_STATE"
        try:
            return ExperienceType(val_str)
        except ValueError:
            raise ValueError(f"Invalid experience type: '{value}'.")

    @model_validator(mode="after")
    def validate_experience_type_consistency(self) -> "ClassificationResult":
        """Enforce fail-closed business rules:
        - When is_experience=False, type MUST be None.
        - When is_experience=True, type MUST be a valid ExperienceType.
        """
        if not self.is_experience and self.type is not None:
            raise ValueError("When is_experience is False, type MUST be None.")
        if self.is_experience and self.type is None:
            raise ValueError("When is_experience is True, type MUST be a valid ExperienceType.")
        return self
