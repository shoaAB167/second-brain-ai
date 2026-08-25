from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator

from personal_ai.domain.experience.enums import ExperienceType


class ExperienceExtractionResult(BaseModel):
    """Structured result produced by the AI Experience Extractor.

    Represents canonical concise personal information extracted from a classified user message.
    Fails closed on missing/empty content or invalid confidence scores.
    Does NOT directly create or modify Experience records in the database.
    """

    content: str = Field(
        ...,
        description="Canonical concise representation of the user-specific experience.",
    )
    type: ExperienceType = Field(
        ...,
        description="Taxonomy category of the extracted experience.",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Categorical domain (e.g. 'career', 'work', 'fitness', 'projects').",
    )
    status: Optional[str] = Field(
        default="active",
        description="Lifecycle status of the experience (e.g. 'active').",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extractor confidence score between 0.0 and 1.0 that structured content reflects user message.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional short explanation of extractor reasoning.",
    )
    raw_model: Optional[str] = Field(
        default=None,
        description="Identifier of the LLM model that generated the extraction.",
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value: Any) -> str:
        """Validate content is non-empty string."""
        if value is None or not str(value).strip():
            raise ValueError("Extracted experience content cannot be empty or whitespace-only.")
        return str(value).strip()

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, value: Any) -> float:
        """Validate confidence is strictly a numeric float/int between 0.0 and 1.0 (not bool/str)."""
        if type(value) is bool or not isinstance(value, (int, float)):
            raise ValueError(f"Confidence score MUST be a numeric float/int, got {type(value).__name__}: {value!r}")
        val_float = float(value)
        if val_float < 0.0 or val_float > 1.0:
            raise ValueError(f"Confidence score '{value}' must be bounded between 0.0 and 1.0 inclusive.")
        return val_float

    @field_validator("type", mode="before")
    @classmethod
    def validate_type_enum(cls, value: Any) -> ExperienceType:
        """Convert string to ExperienceType enum. Fails closed on invalid strings."""
        if value is None or value == "":
            raise ValueError("Experience type is required for extraction.")
        if isinstance(value, ExperienceType):
            return value
        val_str = str(value).upper().strip()
        if val_str == "EMOTION":
            val_str = "EMOTION_STATE"
        try:
            return ExperienceType(val_str)
        except ValueError:
            raise ValueError(f"Invalid experience type: '{value}'.")
