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


class LLMModel(str, Enum):
    """Common model reference constants across supported providers."""

    # OpenAI
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    O1 = "o1"
    O3_MINI = "o3-mini"

    # Anthropic
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"

    # Gemini
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"

    # DeepSeek
    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_REASONER = "deepseek-reasoner"

    # Ollama
    LLAMA_3 = "llama3"
    LLAMA_3_2 = "llama3.2"
    MISTRAL = "mistral"


class LLMResponse(BaseModel):
    """Container for complete LLM response data and execution metadata."""

    content: str = Field(..., description="The generated response text from the LLM.")
    provider: str = Field(..., description="The provider used for text generation.")
    model: str = Field(..., description="The specific model used for generation.")
    latency_ms: float = Field(..., description="Response latency in milliseconds.")
    prompt_tokens: Optional[int] = Field(None, description="Number of tokens in the prompt.")
    completion_tokens: Optional[int] = Field(None, description="Number of tokens in the completion.")
    total_tokens: Optional[int] = Field(None, description="Total tokens consumed.")
