from enum import Enum
from typing import Optional

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

    role: str = Field(..., description="Role of the message sender (e.g. user, assistant, system).")
    content: str = Field(..., description="Text content of the message.")


class LLMResponse(BaseModel):
    """Container for complete LLM response data and execution metadata."""

    content: str = Field(..., description="The generated response text from the LLM.")
    provider: str = Field(..., description="The provider used for text generation.")
    model: str = Field(..., description="The specific model used for generation.")
    latency_ms: float = Field(..., description="Response latency in milliseconds.")
    prompt_tokens: Optional[int] = Field(None, description="Number of tokens in the prompt.")
    completion_tokens: Optional[int] = Field(None, description="Number of tokens in the completion.")
    total_tokens: Optional[int] = Field(None, description="Total tokens consumed.")
