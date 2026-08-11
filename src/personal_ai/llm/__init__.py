from personal_ai.llm.client import LLMClient
from personal_ai.llm.exceptions import (
    LLMAuthenticationException,
    LLMConnectionException,
    LLMException,
    LLMRateLimitException,
)
from personal_ai.llm.litellm_client import LiteLLMClient
from personal_ai.llm.models import LLMModel, LLMProvider, LLMResponse


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
    "LLMProvider",
    "LLMModel",
    "LLMResponse",
    "LLMException",
    "LLMAuthenticationException",
    "LLMRateLimitException",
    "LLMConnectionException",
]
