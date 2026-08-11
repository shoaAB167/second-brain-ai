import time
from typing import Any, Dict, List, Optional

import litellm

from personal_ai.config.settings import Settings, get_settings
from personal_ai.core.logger import get_logger
from personal_ai.llm.client import LLMClient
from personal_ai.llm.exceptions import (
    LLMAuthenticationException,
    LLMConnectionException,
    LLMException,
    LLMRateLimitException,
)

logger = get_logger(__name__)


class LiteLLMClient(LLMClient):
    """LiteLLM implementation of the abstract LLMClient interface.

    Handles provider routing, configuration resolution, exception translation,
    and privacy-preserving execution logging for supported backends (OpenAI,
    Anthropic, Gemini, DeepSeek, OpenRouter, Ollama).
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize the LiteLLM client with settings or explicit overrides.

        Args:
            provider: LLM provider name override. Defaults to settings.llm_provider.
            model: LLM model name override. Defaults to settings.llm_model.
            api_key: Optional API key override. Resolved from settings if omitted.
            settings: Settings instance. Defaults to application settings.
        """
        self._settings = settings or get_settings()
        self._provider = (provider or self._settings.llm_provider).lower()
        self._model = model or self._settings.llm_model
        self._explicit_api_key = api_key
        self._api_key = self._resolve_api_key()

    def _resolve_api_key(self) -> Optional[str]:
        """Resolve API key based on provider and settings."""
        if self._explicit_api_key:
            return self._explicit_api_key

        provider = self._provider.lower()
        if provider == "openai":
            return self._settings.openai_api_key
        elif provider == "anthropic":
            return self._settings.anthropic_api_key
        elif provider in ("gemini", "google"):
            return self._settings.google_api_key
        elif provider == "deepseek":
            return self._settings.deepseek_api_key
        elif provider == "openrouter":
            return self._settings.openrouter_api_key
        return None

    def _format_model_name(self) -> str:
        """Format the model identifier expected by LiteLLM."""
        if "/" in self._model:
            return self._model
        return f"{self._provider}/{self._model}"

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text response using LiteLLM.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system instruction message.
            **kwargs: Extra parameters passed to completion engine.

        Returns:
            str: Generated text response.

        Raises:
            LLMAuthenticationException: On authentication errors.
            LLMRateLimitException: On quota/rate limit errors.
            LLMConnectionException: On network/timeout errors.
            LLMException: On general execution errors.
        """
        model_name = self._format_model_name()
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_kwargs: Dict[str, Any] = dict(kwargs)
        request_kwargs["model"] = model_name
        request_kwargs["messages"] = messages

        if self._api_key:
            request_kwargs["api_key"] = self._api_key

        if self._provider == "ollama" and self._settings.ollama_api_base:
            request_kwargs.setdefault("api_base", self._settings.ollama_api_base)

        logger.info(
            "Sending LLM request [provider=%s, model=%s]",
            self._provider,
            self._model,
        )

        start_time = time.perf_counter()
        try:
            response = await litellm.acompletion(**request_kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "LLM request completed [provider=%s, model=%s, latency=%.2fms]",
                self._provider,
                self._model,
                latency_ms,
            )

            content = response.choices[0].message.content
            return content if content is not None else ""

        except litellm.exceptions.AuthenticationError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM authentication failed [provider=%s, model=%s, latency=%.2fms]",
                self._provider,
                self._model,
                latency_ms,
            )
            raise LLMAuthenticationException(
                message=f"LLM authentication failed for provider '{self._provider}': {exc}",
                details={"provider": self._provider, "model": self._model},
            ) from exc

        except litellm.exceptions.RateLimitError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM rate limit exceeded [provider=%s, model=%s, latency=%.2fms]",
                self._provider,
                self._model,
                latency_ms,
            )
            raise LLMRateLimitException(
                message=f"LLM rate limit exceeded for provider '{self._provider}': {exc}",
                details={"provider": self._provider, "model": self._model},
            ) from exc

        except (
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.Timeout,
            litellm.exceptions.ServiceUnavailableError,
        ) as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM connection error [provider=%s, model=%s, latency=%.2fms]",
                self._provider,
                self._model,
                latency_ms,
            )
            raise LLMConnectionException(
                message=f"LLM connection error for provider '{self._provider}': {exc}",
                details={"provider": self._provider, "model": self._model},
            ) from exc

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM request failed [provider=%s, model=%s, latency=%.2fms]: %s",
                self._provider,
                self._model,
                latency_ms,
                exc,
            )
            raise LLMException(
                message=f"LLM operation failed for provider '{self._provider}': {exc}",
                details={"provider": self._provider, "model": self._model},
            ) from exc

    async def health_check(self) -> bool:
        """Verify client reachability to the configured provider.

        Returns:
            bool: True if the provider is responsive, False otherwise.
        """
        try:
            await self.generate_response(prompt="ping", max_tokens=1)
            return True
        except Exception as exc:
            logger.warning(
                "LLM health check failed [provider=%s, model=%s]: %s",
                self._provider,
                self._model,
                exc,
            )
            return False
