from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    """Supported LLM provider identifiers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


class LLMMessage(BaseModel):
    """Domain message representation for LLM interactions."""

    role: str = Field(..., description="Role of the message sender (e.g. user, assistant, system, tool).")
    content: str = Field(..., description="Text content of the message.")


class ToolCall(BaseModel):
    """Domain model representing a structured tool call requested by an LLM."""

    id: Optional[str] = Field(default=None, description="Optional unique tool call identifier.")
    name: str = Field(..., description="Name of the tool capability to invoke.")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Parsed arguments dictionary for the tool."
    )


class LLMResponse(BaseModel):
    """Container for complete LLM response data and execution metadata."""

    content: str = Field(..., description="The generated response text from the LLM.")
    provider: str = Field(..., description="The provider used for text generation.")
    model: str = Field(..., description="The specific model used for generation.")
    latency_ms: float = Field(..., description="Response latency in milliseconds.")
    prompt_tokens: Optional[int] = Field(None, description="Number of tokens in the prompt.")
    completion_tokens: Optional[int] = Field(None, description="Number of tokens in the completion.")
    total_tokens: Optional[int] = Field(None, description="Total tokens consumed.")
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None, description="Optional structured tool calls requested by the LLM."
    )


class LLMStreamChunk(BaseModel):
    """Domain model representing a single streamed chunk from an LLM provider."""

    content: str = Field(..., description="The partial text content of this chunk.")
    finish_reason: Optional[str] = Field(
        default=None, description="Completion finish reason (e.g. 'stop', 'length') if finished."
    )
    usage: Optional[Dict[str, int]] = Field(
        default=None, description="Optional token usage metrics if provided in final chunk."
    )
