from typing import Optional
import uuid

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Schema for incoming chat API request."""

    message: str = Field(..., min_length=1, description="The user prompt or message.")
    conversation_id: Optional[uuid.UUID] = Field(
        default=None, description="Optional UUID of an existing conversation thread."
    )
    system_prompt: Optional[str] = Field(
        default=None, description="Optional system instruction for guiding model response."
    )


class ChatResponse(BaseModel):
    """Schema for API chat response."""

    conversation_id: uuid.UUID = Field(..., description="The conversation UUID.")
    response: str = Field(..., description="The generated text response from the LLM.")
    provider: str = Field(..., description="The LLM provider used.")
    model: str = Field(..., description="The LLM model used.")
    latency_ms: float = Field(..., description="Execution latency in milliseconds.")
    prompt_tokens: Optional[int] = Field(default=None, description="Prompt tokens used.")
    completion_tokens: Optional[int] = Field(default=None, description="Completion tokens used.")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens used.")
