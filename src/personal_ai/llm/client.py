from abc import ABC, abstractmethod
from typing import Any, Optional


class LLMClient(ABC):
    """Abstract client interface for provider-independent LLM interactions.

    All LLM backends (OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter, Ollama)
    must implement this interface. The rest of the application interacts strictly
    with this abstraction.
    """

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text response from the configured LLM backend.

        Args:
            prompt: The input user prompt or text message.
            system_prompt: Optional system instruction to guide model output.
            **kwargs: Optional execution hyper-parameters (e.g., temperature, max_tokens).

        Returns:
            str: The generated text content from the LLM.

        Raises:
            LLMException: Base exception for any LLM generation failure.
            LLMAuthenticationException: If authentication credentials are invalid.
            LLMRateLimitException: If quota or rate limits are exceeded.
            LLMConnectionException: If connection or network issues occur.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Perform a health check to verify connectivity with the configured LLM provider.

        Returns:
            bool: True if the provider is reachable and responsive, False otherwise.
        """
        pass
