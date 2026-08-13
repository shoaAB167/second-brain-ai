from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List

from personal_ai.llm.models import LLMMessage, LLMResponse, LLMStreamChunk


class LLMClient(ABC):
    """Abstract client interface for provider-independent LLM interactions.

    All LLM backends (OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter, Ollama)
    must implement this interface. The rest of the application interacts strictly
    with this abstraction.
    """

    @abstractmethod
    async def generate_response(
        self,
        messages: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a text response given a sequence of domain LLMMessage objects.

        Args:
            messages: List of structured domain messages (system, user, assistant).
            **kwargs: Optional execution hyper-parameters (e.g., temperature, max_tokens).

        Returns:
            LLMResponse: The structured LLM response containing text content and metadata.

        Raises:
            LLMException: Base exception for any LLM generation failure.
            LLMAuthenticationException: If authentication credentials are invalid.
            LLMRateLimitException: If quota or rate limits are exceeded.
            LLMConnectionException: If connection or network issues occur.
        """
        pass

    @abstractmethod
    def stream_response(
        self,
        messages: List[LLMMessage],
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream response chunks from the LLM provider as an async iterator.

        Args:
            messages: List of structured domain messages (system, user, assistant).
            **kwargs: Optional execution hyper-parameters (e.g., temperature, max_tokens).

        Returns:
            AsyncIterator[LLMStreamChunk]: Async iterator yielding domain chunk objects.

        Raises:
            LLMException: Base exception for any LLM generation failure.
            LLMAuthenticationException: If authentication credentials are invalid.
            LLMRateLimitException: If quota or rate limits are exceeded.
            LLMConnectionException: If connection or network issues occur.
        """
        pass
