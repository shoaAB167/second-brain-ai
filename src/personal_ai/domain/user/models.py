from datetime import datetime
import uuid

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """Schema for user registration request."""

    email: EmailStr = Field(..., description="User's unique email address.")
    password: str = Field(
        ...,
        min_length=8,
        description="Plaintext password (minimum 8 characters).",
        json_schema_extra={"example": "securepassword123"},
    )


class LoginRequest(BaseModel):
    """Schema for user authentication/login request."""

    email: EmailStr = Field(..., description="User's email address.")
    password: str = Field(..., description="Plaintext password.")


class UserResponse(BaseModel):
    """Safe schema for API user responses (never returns password_hash)."""

    id: uuid.UUID = Field(..., description="User's unique UUID.")
    email: EmailStr = Field(..., description="User's email address.")
    created_at: datetime = Field(..., description="Timestamp when user registered.")


class TokenResponse(BaseModel):
    """Schema for JWT access token response."""

    access_token: str = Field(..., description="JWT access token string.")
    token_type: str = Field(default="bearer", description="Token type header.")
