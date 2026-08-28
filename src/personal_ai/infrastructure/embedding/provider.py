from abc import ABC, abstractmethod
import math
from typing import List, Optional
import httpx

from personal_ai.config.settings import get_settings
from personal_ai.core.logger import get_logger
from personal_ai.llm.exceptions import (
    LLMAuthenticationException,
    LLMConnectionException,
    LLMException,
    LLMRateLimitException,
)

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    """Abstract port interface for vector embedding generation."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier name."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return embedding model identifier."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return target vector embedding dimension size."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate a vector embedding for the input text."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Infrastructure implementation of EmbeddingProvider using OpenAI's Embeddings API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> None:
        """Initialize OpenAI embedding provider.

        Args:
            api_key: Optional OpenAI API key override.
            model: Optional model identifier override.
            dimensions: Optional vector dimensions size override.
        """
        settings = get_settings()
        self._api_key = api_key or settings.openai_api_key
        self._model = model or settings.embedding_model
        self._dimensions = dimensions or settings.embedding_dimensions

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> List[float]:
        """Generate vector embedding via OpenAI REST API endpoint."""
        if not self._api_key:
            raise LLMAuthenticationException("OpenAI API key is not configured for vector embeddings.")

        if not text or not text.strip():
            raise ValueError("Embedding input text cannot be empty.")

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": text,
            "model": self._model,
            "dimensions": self._dimensions,
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 401:
                    raise LLMAuthenticationException("Invalid OpenAI API Key.")
                elif response.status_code == 429:
                    raise LLMRateLimitException("OpenAI API rate limit exceeded during embedding generation.")
                elif response.status_code != 200:
                    raise LLMException(f"OpenAI API returned status code {response.status_code}: {response.text}")

                data = response.json()
                vector = data["data"][0]["embedding"]
                if len(vector) != self._dimensions:
                    logger.warning(
                        "Returned vector dimension mismatch [expected=%d, got=%d]",
                        self._dimensions,
                        len(vector),
                    )
                return [float(x) for x in vector]

        except httpx.RequestError as exc:
            raise LLMConnectionException(f"Network error connecting to OpenAI Embedding API: {exc}")


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock implementation of EmbeddingProvider for offline tests and local simulation."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        should_fail: bool = False,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self.should_fail = should_fail

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> List[float]:
        if self.should_fail:
            raise LLMConnectionException("Mock embedding provider connection failure.")

        if not text or not text.strip():
            raise ValueError("Embedding input text cannot be empty.")

        # Generate deterministic unit-normalized float vector based on text hash
        seed = sum(ord(c) for c in text)
        raw_vector = [(math.sin(seed + i)) for i in range(self._dimensions)]
        norm = math.sqrt(sum(x * x for x in raw_vector)) or 1.0
        return [x / norm for x in raw_vector]
