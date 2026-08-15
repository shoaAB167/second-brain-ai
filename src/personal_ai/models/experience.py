from typing import Optional
import uuid

from pydantic import BaseModel, Field, field_validator


class RecordExperienceRequest(BaseModel):
    """Schema for recording a new raw Experience."""

    content: str = Field(..., description="Raw, uninterpreted user experience observation text.")
    source: str = Field(..., description="Source system/channel (e.g. CHAT, FILE).")

    @field_validator("content")
    @classmethod
    def validate_content_non_empty(cls, value: str) -> str:
        """Validate content is non-empty and not whitespace-only."""
        if not isinstance(value, str) or not value or not value.strip():
            raise ValueError("Experience content cannot be empty or whitespace-only.")
        return value

    @field_validator("source")
    @classmethod
    def validate_source_uppercase(cls, value: str) -> str:
        """Validate source is non-empty string."""
        if not isinstance(value, str) or not value or not value.strip():
            raise ValueError("Experience source cannot be empty.")
        return value.strip().upper()


class RecordExperienceResponse(BaseModel):
    """Response payload for HTTP 201 Created Experience creation."""

    experience_id: uuid.UUID = Field(
        ...,
        alias="experienceId",
        description="UUID of the recorded experience.",
    )
    status: str = Field(default="RECEIVED", description="Status of experience (RECEIVED).")
    message: str = Field(
        default="Experience recorded successfully.",
        description="User-facing confirmation message.",
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "experienceId": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "status": "RECEIVED",
                "message": "Experience recorded successfully.",
            }
        },
    }
