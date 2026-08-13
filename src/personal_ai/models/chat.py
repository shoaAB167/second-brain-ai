from enum import Enum
from typing import Optional
import uuid

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Schema for incoming chat API request."""

    message: str = Field(
        ...,
        min_length=1,
        description="The user prompt or message.",
        json_schema_extra={"example": "Hello"},
    )
    conversation_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Optional UUID of an existing conversation thread. Omit to start a new conversation.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system instruction for guiding model response.",
    )


class ChatResponse(BaseModel):
    """Schema for API chat response."""

    conversation_id: uuid.UUID = Field(..., description="The conversation UUID.")
    response: str = Field(..., description="The generated text response from the LLM.")
    provider: str = Field(..., description="The LLM provider used.")
    model: str = Field(..., description="The specific model used.")
    latency_ms: float = Field(..., description="Execution latency in milliseconds.")
    prompt_tokens: Optional[int] = Field(default=None, description="Prompt tokens used.")
    completion_tokens: Optional[int] = Field(default=None, description="Completion tokens used.")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens used.")


class StreamEventType(str, Enum):
    """Event types for Server-Sent Events (SSE) streaming."""

    TOKEN = "token"
    DONE = "done"
    ERROR = "error"


class ChatStreamEvent(BaseModel):
    """Schema for individual SSE streaming events sent to client."""

    type: StreamEventType = Field(..., description="Event type: token, done, or error.")
    content: Optional[str] = Field(default=None, description="Text chunk for token event.")
    message: Optional[str] = Field(default=None, description="Error message for error event.")

    def to_sse(self) -> str:
        """Format event model as a Server-Sent Events (SSE) data line."""
        return f"data: {self.model_dump_json(exclude_none=True)}\n\n"
