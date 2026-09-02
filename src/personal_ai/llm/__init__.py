from personal_ai.llm.client import LLMClient
from personal_ai.llm.exceptions import (
    LLMAuthenticationException,
    LLMConnectionException,
    LLMException,
    LLMRateLimitException,
    LLMServiceUnavailableException,
    LLMTimeoutException,
)
from personal_ai.llm.litellm_client import LiteLLMClient
from personal_ai.llm.models import LLMMessage, LLMProvider, LLMResponse, LLMStreamChunk


def get_llm_client() -> LLMClient:
    """Dependency injection factory providing an abstract LLMClient instance.

    The client implementation is dynamically constructed from application settings.

    Returns:
        LLMClient: Configured abstract LLM client instance.
    """
    return LiteLLMClient()


__all__ = [
    "LLMClient",
    "get_llm_client",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "LLMStreamChunk",
    "LLMException",
    "LLMAuthenticationException",
    "LLMRateLimitException",
    "LLMConnectionException",
    "LLMTimeoutException",
    "LLMServiceUnavailableException",
]
