import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import litellm
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from personal_ai.config.settings import Settings, get_settings
from personal_ai.core.logger import get_logger
from personal_ai.llm.client import LLMClient
from personal_ai.llm.exceptions import (
    LLMAuthenticationException,
    LLMConnectionException,
    LLMException,
    LLMRateLimitException,
)
from personal_ai.llm.models import LLMMessage, LLMResponse, LLMStreamChunk

logger = get_logger(__name__)

# Suppress verbose LiteLLM logging by default
litellm.suppress_debug_info = True


class LiteLLMClient(LLMClient):
    """LiteLLM-backed implementation of LLMClient interface.

    Handles provider normalization, API key resolution, and exception translation.
    Isolated within personal_ai.llm module.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize LiteLLMClient with optional parameter overrides."""
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
            return self._settings.gemini_api_key or self._settings.google_api_key
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
        messages: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text response using LiteLLM."""
        model_name = self._format_model_name()
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        request_kwargs: Dict[str, Any] = dict(kwargs)
        request_kwargs["model"] = model_name
        request_kwargs["messages"] = formatted_messages

        if self._api_key:
            request_kwargs["api_key"] = self._api_key

        if self._provider == "ollama" and self._settings.ollama_api_base:
            request_kwargs.setdefault("api_base", self._settings.ollama_api_base)

        logger.info(
            "Sending LLM request [provider=%s, model=%s, messages_count=%d]",
            self._provider,
            self._model,
            len(messages),
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

            content = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)

            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
            total_tokens = getattr(usage, "total_tokens", None) if usage else None

            return LLMResponse(
                content=content,
                provider=self._provider,
                model=self._model,
                latency_ms=round(latency_ms, 2),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        except AuthenticationError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM authentication failed [provider=%s, model=%s, latency=%.2fms]: %s",
                self._provider,
                self._model,
                latency_ms,
                exc,
            )
            raise LLMAuthenticationException(
                message="LLM provider authentication failed. Please check configured API key.",
                details={"provider": self._provider, "model": self._model},
            ) from exc

        except RateLimitError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM rate limit exceeded [provider=%s, model=%s, latency=%.2fms]: %s",
                self._provider,
                self._model,
                latency_ms,
                exc,
            )
            raise LLMRateLimitException(
                message="LLM provider rate limit exceeded. Please try again later.",
                details={"provider": self._provider, "model": self._model},
            ) from exc

        except (
            APIConnectionError,
            Timeout,
            ServiceUnavailableError,
        ) as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM connection error [provider=%s, model=%s, latency=%.2fms]: %s",
                self._provider,
                self._model,
                latency_ms,
                exc,
            )
            raise LLMConnectionException(
                message="Failed to connect to LLM provider. Please try again later.",
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
                message="An error occurred while processing the LLM request.",
                details={"provider": self._provider, "model": self._model},
            ) from exc

    async def stream_response(
        self,
        messages: List[LLMMessage],
        **kwargs: Any,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Stream text response chunks using LiteLLM.

        Args:
            messages: List of domain LLMMessage objects.
            **kwargs: Extra parameters passed to completion engine.

        Yields:
            LLMStreamChunk: Domain chunk objects containing partial text content.

        Raises:
            LLMAuthenticationException: On authentication errors.
            LLMRateLimitException: On quota/rate limit errors.
            LLMConnectionException: On network/timeout errors.
            LLMException: On general execution errors.
        """
        model_name = self._format_model_name()
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        request_kwargs: Dict[str, Any] = dict(kwargs)
        request_kwargs["model"] = model_name
        request_kwargs["messages"] = formatted_messages
        request_kwargs["stream"] = True

        if self._api_key:
            request_kwargs["api_key"] = self._api_key

        if self._provider == "ollama" and self._settings.ollama_api_base:
            request_kwargs.setdefault("api_base", self._settings.ollama_api_base)

        logger.info(
            "Sending LLM stream request [provider=%s, model=%s, messages_count=%d]",
            self._provider,
            self._model,
            len(messages),
        )

        start_time = time.perf_counter()
        try:
            response = await litellm.acompletion(**request_kwargs)
            async for chunk in response:
                delta_content = ""
                finish_reason = None

                if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta:
                        delta_content = getattr(delta, "content", "") or ""
                    elif hasattr(choice, "text"):
                        delta_content = getattr(choice, "text", "") or ""

                    finish_reason = getattr(choice, "finish_reason", None)

                usage_dict = None
                raw_usage = getattr(chunk, "usage", None)
                if raw_usage:
                    usage_dict = {
                        "prompt_tokens": getattr(raw_usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(raw_usage, "completion_tokens", 0),
                        "total_tokens": getattr(raw_usage, "total_tokens", 0),
                    }

                yield LLMStreamChunk(
                    content=delta_content,
                    finish_reason=finish_reason,
                    usage=usage_dict,
                )

            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "LLM stream completed [provider=%s, model=%s, latency=%.2fms]",
                self._provider,
                self._model,
                latency_ms,
            )

        except AuthenticationError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM stream authentication failed [provider=%s, model=%s, latency=%.2fms]: %s",
                self._provider,
                self._model,
                latency_ms,
                exc,
            )
            raise LLMAuthenticationException(
                message="LLM provider authentication failed. Please check configured API key.",
                details={"provider": self._provider, "model": self._model},
            ) from exc

        except RateLimitError as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM stream rate limit exceeded [provider=%s, model=%s, latency=%.2fms]: %s",
                self._provider,
                self._model,
                latency_ms,
                exc,
            )
            raise LLMRateLimitException(
                message="LLM provider rate limit exceeded. Please try again later.",
                details={"provider": self._provider, "model": self._model},
            ) from exc

        except (
            APIConnectionError,
            Timeout,
            ServiceUnavailableError,
        ) as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM stream connection error [provider=%s, model=%s, latency=%.2fms]: %s",
                self._provider,
                self._model,
                latency_ms,
                exc,
            )
            raise LLMConnectionException(
                message="Failed to connect to LLM provider. Please try again later.",
                details={"provider": self._provider, "model": self._model},
            ) from exc

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "LLM stream request failed [provider=%s, model=%s, latency=%.2fms]: %s",
                self._provider,
                self._model,
                latency_ms,
                exc,
            )
            raise LLMException(
                message="An error occurred while processing the LLM request.",
                details={"provider": self._provider, "model": self._model},
            ) from exc
