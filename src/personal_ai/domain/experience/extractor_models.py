from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ExperienceExtractionResult(BaseModel):
    """Structured result produced by the AI Experience Extractor.

    Represents canonical concise personal information extracted from a classified user message.
    Explicitly tracks success vs failure. Fails closed on invalid fields or missing content.
    Does NOT directly create or modify Experience records in the database.
    """

    success: bool = Field(
        ...,
        description="True if structured extraction succeeded, False if extraction failed.",
    )
    content: Optional[str] = Field(
        default=None,
        description="Canonical concise representation of the user-specific experience.",
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

    @field_validator("success", mode="before")
    @classmethod
    def validate_strict_bool(cls, value: Any) -> bool:
        """Validate that success is strictly a JSON boolean (True/False)."""
        if type(value) is not bool:
            raise ValueError(f"success MUST be a strict boolean (true/false), got {type(value).__name__}: {value!r}")
        return value

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

    @field_validator("content", mode="before")
    @classmethod
    def validate_content_string(cls, value: Any) -> Optional[str]:
        """Validate content string. Empty or whitespace-only values are normalized to None."""
        if value is None:
            return None
        val_str = str(value).strip()
        return val_str if val_str else None

    @model_validator(mode="after")
    def validate_success_consistency(self) -> "ExperienceExtractionResult":
        """Enforce fail-closed business rules:
        - When success=True, content MUST be a non-empty string.
        - When success=False, content MUST be None.
        """
        if self.success and not self.content:
            raise ValueError("When extraction success is True, content MUST be a non-empty string.")
        if not self.success and self.content is not None:
            raise ValueError("When extraction success is False, content MUST be None.")
        return self
