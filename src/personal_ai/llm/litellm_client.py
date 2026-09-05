import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadGatewayError,
    InternalServerError,
    MidStreamFallbackError,
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
    LLMServiceUnavailableException,
    LLMTimeoutException,
)
from personal_ai.llm.models import LLMMessage, LLMResponse, LLMStreamChunk, ToolCall

logger = get_logger(__name__)

# Suppress verbose LiteLLM logging by default
litellm.suppress_debug_info = True


class LiteLLMClient(LLMClient):
    """LiteLLM-backed implementation of LLMClient interface.

    Handles provider normalization, API key resolution, bounded timeouts,
    exponential backoff retries for transient provider failures, and exception translation.
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

        # Resilience settings
        self._request_timeout: float = getattr(self._settings, "llm_request_timeout", 30.0)
        self._stream_start_timeout: float = getattr(self._settings, "llm_stream_start_timeout", 30.0)
        self._stream_chunk_timeout: float = getattr(self._settings, "llm_stream_chunk_timeout", 30.0)
        self._max_retries: int = getattr(self._settings, "llm_max_retries", 2)
        self._retry_initial_delay: float = getattr(self._settings, "llm_retry_initial_delay", 1.0)
        self._retry_backoff_factor: float = getattr(self._settings, "llm_retry_backoff_factor", 2.0)
        self._retry_max_delay: float = getattr(self._settings, "llm_retry_max_delay", 4.0)

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

    def _is_transient_error(self, exc: Exception) -> bool:
        """Determine if an exception represents a transient failure eligible for retry.

        Classification priority:
        1. Explicit LiteLLM / standard exception types (including MidStreamFallbackError)
        2. HTTP/status code (500, 502, 503, 504)
        3. String matching as a fallback for vendor/beta exceptions
        """
        if isinstance(exc, (AuthenticationError,)):
            return False

        # 1. Explicit LiteLLM / standard exception types
        if isinstance(
            exc,
            (
                ServiceUnavailableError,
                MidStreamFallbackError,
                BadGatewayError,
                InternalServerError,
                APIConnectionError,
                Timeout,
                asyncio.TimeoutError,
            ),
        ):
            return True

        # 2. HTTP status code check
        status_code = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
        if status_code in (500, 502, 503, 504):
            return True

        # Check wrapped original exception if available
        orig_exc = getattr(exc, "original_exception", None)
        if orig_exc is not None and isinstance(orig_exc, Exception):
            if self._is_transient_error(orig_exc):
                return True

        # 3. Fallback string matching for vendor/beta errors (e.g. Vertex_ai_betaException)
        exc_str = str(exc).lower()
        transient_indicators = [
            "503",
            "502",
            "504",
            "high demand",
            "serviceunavailable",
            "temporarily unavailable",
            "resource_exhausted",
            "overloaded",
            "connection error",
            "connection reset",
            "timeout",
            "timed out",
        ]
        return any(indicator in exc_str for indicator in transient_indicators)

    def _map_exception(self, exc: Exception, duration_ms: float = 0.0) -> LLMException:
        """Map provider-specific / LiteLLM / runtime exceptions into domain LLMException hierarchy."""
        if isinstance(exc, (LLMAuthenticationException, LLMRateLimitException, LLMTimeoutException, LLMServiceUnavailableException, LLMConnectionException)):
            return exc

        if isinstance(exc, AuthenticationError):
            return LLMAuthenticationException(
                message="LLM provider authentication failed. Please check configured API key.",
                details={"provider": self._provider, "model": self._model},
            )

        if isinstance(exc, RateLimitError):
            return LLMRateLimitException(
                message="LLM provider rate limit exceeded. Please try again later.",
                details={"provider": self._provider, "model": self._model},
            )

        if isinstance(exc, (Timeout, asyncio.TimeoutError)) or "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
            return LLMTimeoutException(
                message="AI service is temporarily unavailable. Please try again.",
                details={"provider": self._provider, "model": self._model, "reason": "timeout"},
            )

        if (
            isinstance(exc, (ServiceUnavailableError, MidStreamFallbackError))
            or "503" in str(exc)
            or "high demand" in str(exc).lower()
            or "serviceunavailable" in str(exc).lower()
        ):
            return LLMServiceUnavailableException(
                message="AI service is temporarily unavailable. Please try again.",
                details={"provider": self._provider, "model": self._model, "reason": "service_unavailable"},
            )

        if isinstance(exc, (APIConnectionError, BadGatewayError, InternalServerError)) or "connection" in str(exc).lower():
            return LLMConnectionException(
                message="AI service is temporarily unavailable. Please try again.",
                details={"provider": self._provider, "model": self._model},
            )

        # Catch-all general domain LLM exception
        return LLMException(
            message="An error occurred while processing the LLM request.",
            details={"provider": self._provider, "model": self._model},
        )

    async def generate_response(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text response using LiteLLM with bounded timeout and exponential backoff retries."""
        model_name = self._format_model_name()
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        request_kwargs: Dict[str, Any] = dict(kwargs)
        request_kwargs["model"] = model_name
        request_kwargs["messages"] = formatted_messages
        request_kwargs.setdefault("timeout", self._request_timeout)

        if tools:
            request_kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": getattr(t, "name", t.get("name") if isinstance(t, dict) else ""),
                        "description": getattr(t, "description", t.get("description") if isinstance(t, dict) else ""),
                        "parameters": getattr(t, "input_schema", t.get("parameters") if isinstance(t, dict) else {}),
                    },
                }
                for t in tools
            ]

        if self._api_key:
            request_kwargs["api_key"] = self._api_key

        if self._provider == "ollama" and self._settings.ollama_api_base:
            request_kwargs.setdefault("api_base", self._settings.ollama_api_base)

        max_retries = kwargs.get("max_retries", self._max_retries)
        delay = self._retry_initial_delay
        total_attempts = 1 + max_retries

        logger.info(
            "Sending LLM request [provider=%s, model=%s, messages_count=%d, max_retries=%d]",
            self._provider,
            self._model,
            len(messages),
            max_retries,
        )

        overall_start = time.perf_counter()
        last_exception: Optional[Exception] = None

        for attempt in range(1, total_attempts + 1):
            attempt_start = time.perf_counter()
            try:
                response = await asyncio.wait_for(
                    litellm.acompletion(**request_kwargs),
                    timeout=self._request_timeout,
                )
                latency_ms = (time.perf_counter() - attempt_start) * 1000
                total_latency_ms = (time.perf_counter() - overall_start) * 1000

                logger.info(
                    "LLM request completed [provider=%s, model=%s, attempt=%d, latency=%.2fms, total_latency=%.2fms]",
                    self._provider,
                    self._model,
                    attempt,
                    latency_ms,
                    total_latency_ms,
                )

                content = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)

                prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
                completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
                total_tokens = getattr(usage, "total_tokens", None) if usage else None

                raw_tool_calls = getattr(response.choices[0].message, "tool_calls", None)
                parsed_tool_calls: Optional[List[ToolCall]] = None
                if raw_tool_calls:
                    parsed_tool_calls = []
                    for tc in raw_tool_calls:
                        tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)
                        fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else {})
                        fn_name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else "")
                        fn_args = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else {})
                        if isinstance(fn_args, str):
                            try:
                                parsed_args = json.loads(fn_args)
                            except Exception:
                                parsed_args = {}
                        elif isinstance(fn_args, dict):
                            parsed_args = fn_args
                        else:
                            parsed_args = {}
                        parsed_tool_calls.append(
                            ToolCall(id=tc_id, name=fn_name, arguments=parsed_args)
                        )

                return LLMResponse(
                    content=content,
                    provider=self._provider,
                    model=self._model,
                    latency_ms=round(latency_ms, 2),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    tool_calls=parsed_tool_calls,
                )

            except Exception as exc:
                last_exception = exc
                attempt_duration_ms = (time.perf_counter() - attempt_start) * 1000
                is_transient = self._is_transient_error(exc)

                if is_transient and attempt < total_attempts:
                    logger.warning(
                        "Transient LLM request failure [provider=%s, model=%s, attempt=%d/%d, duration=%.2fms]: %s. Retrying in %.2fs...",
                        self._provider,
                        self._model,
                        attempt,
                        total_attempts,
                        attempt_duration_ms,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * self._retry_backoff_factor, self._retry_max_delay)
                else:
                    total_duration_ms = (time.perf_counter() - overall_start) * 1000
                    logger.error(
                        "LLM request failed [provider=%s, model=%s, attempts=%d, total_duration=%.2fms]: %s",
                        self._provider,
                        self._model,
                        attempt,
                        total_duration_ms,
                        exc,
                    )
                    raise self._map_exception(exc, duration_ms=total_duration_ms) from exc

        total_duration_ms = (time.perf_counter() - overall_start) * 1000
        raise self._map_exception(last_exception or Exception("Unknown LLM error"), duration_ms=total_duration_ms)

    async def stream_response(
        self,
        messages: List[LLMMessage],
        **kwargs: Any,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        """Stream text response chunks using LiteLLM with resilience, bounded timeouts, and retries.

        Args:
            messages: List of domain LLMMessage objects.
            **kwargs: Extra parameters passed to completion engine.

        Yields:
            LLMStreamChunk: Domain chunk objects containing partial text content.

        Raises:
            LLMAuthenticationException: On authentication errors.
            LLMRateLimitException: On quota/rate limit errors.
            LLMConnectionException: On network/timeout/service unavailable errors.
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
        request_kwargs.setdefault("timeout", self._request_timeout)

        if self._api_key:
            request_kwargs["api_key"] = self._api_key

        if self._provider == "ollama" and self._settings.ollama_api_base:
            request_kwargs.setdefault("api_base", self._settings.ollama_api_base)

        max_retries = kwargs.get("max_retries", self._max_retries)
        delay = self._retry_initial_delay
        total_attempts = 1 + max_retries

        logger.info(
            "Sending LLM stream request [provider=%s, model=%s, messages_count=%d, max_retries=%d]",
            self._provider,
            self._model,
            len(messages),
            max_retries,
        )

        overall_start = time.perf_counter()
        stream_iter = None
        first_chunk = None
        chunks_count = 0

        # Phase 1: Establish stream and fetch first chunk with bounded retries
        for attempt in range(1, total_attempts + 1):
            attempt_start = time.perf_counter()
            try:
                # 1. Initiate stream request with bounded start timeout
                response = await asyncio.wait_for(
                    litellm.acompletion(**request_kwargs),
                    timeout=self._stream_start_timeout,
                )

                # 2. Extract async iterator
                if hasattr(response, "__aiter__"):
                    stream_iter = response.__aiter__()
                elif hasattr(response, "__iter__"):
                    async def _wrap_sync(sync_iter: Any) -> AsyncGenerator[Any, None]:
                        for item in sync_iter:
                            yield item
                    stream_iter = _wrap_sync(response)
                else:
                    stream_iter = response

                # 3. Wait for first chunk with bounded stream_start_timeout
                first_chunk = await asyncio.wait_for(
                    stream_iter.__anext__(),
                    timeout=self._stream_start_timeout,
                )

                ttft_ms = (time.perf_counter() - overall_start) * 1000
                logger.info(
                    "LLM stream first token received [provider=%s, model=%s, attempt=%d, ttft=%.2fms]",
                    self._provider,
                    self._model,
                    attempt,
                    ttft_ms,
                )
                break

            except StopAsyncIteration:
                # Stream ended immediately with no chunks
                break

            except Exception as exc:
                attempt_duration_ms = (time.perf_counter() - attempt_start) * 1000
                is_transient = self._is_transient_error(exc)

                if is_transient and attempt < total_attempts:
                    logger.warning(
                        "Transient LLM stream failure before first token [provider=%s, model=%s, attempt=%d/%d, duration=%.2fms]: %s. Retrying in %.2fs...",
                        self._provider,
                        self._model,
                        attempt,
                        total_attempts,
                        attempt_duration_ms,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * self._retry_backoff_factor, self._retry_max_delay)
                else:
                    total_duration_ms = (time.perf_counter() - overall_start) * 1000
                    logger.error(
                        "LLM stream failed before first token [provider=%s, model=%s, attempts=%d, total_duration=%.2fms]: %s",
                        self._provider,
                        self._model,
                        attempt,
                        total_duration_ms,
                        exc,
                    )
                    raise self._map_exception(exc, duration_ms=total_duration_ms) from exc

        # Phase 2: Yield first chunk and stream subsequent chunks with per-chunk timeout
        def _parse_chunk(chunk_obj: Any) -> LLMStreamChunk:
            delta_content = ""
            finish_reason = None

            if hasattr(chunk_obj, "choices") and len(chunk_obj.choices) > 0:
                choice = chunk_obj.choices[0]
                delta = getattr(choice, "delta", None)
                if delta:
                    delta_content = getattr(delta, "content", "") or ""
                elif hasattr(choice, "text"):
                    delta_content = getattr(choice, "text", "") or ""

                finish_reason = getattr(choice, "finish_reason", None)

            usage_dict = None
            raw_usage = getattr(chunk_obj, "usage", None)
            if raw_usage:
                usage_dict = {
                    "prompt_tokens": getattr(raw_usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(raw_usage, "completion_tokens", 0),
                    "total_tokens": getattr(raw_usage, "total_tokens", 0),
                }

            return LLMStreamChunk(
                content=delta_content,
                finish_reason=finish_reason,
                usage=usage_dict,
            )

        if stream_iter is not None:
            try:
                # Yield first chunk if retrieved
                if first_chunk is not None:
                    chunks_count += 1
                    yield _parse_chunk(first_chunk)

                # Stream remaining chunks
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            stream_iter.__anext__(),
                            timeout=self._stream_chunk_timeout,
                        )
                        chunks_count += 1
                        yield _parse_chunk(chunk)
                    except StopAsyncIteration:
                        break

                total_duration_ms = (time.perf_counter() - overall_start) * 1000
                logger.info(
                    "LLM stream completed [provider=%s, model=%s, total_duration=%.2fms, chunks_count=%d]",
                    self._provider,
                    self._model,
                    total_duration_ms,
                    chunks_count,
                )

            except Exception as exc:
                total_duration_ms = (time.perf_counter() - overall_start) * 1000
                logger.error(
                    "LLM stream failed mid-stream [provider=%s, model=%s, chunks_yielded=%d, duration=%.2fms]: %s",
                    self._provider,
                    self._model,
                    chunks_count,
                    total_duration_ms,
                    exc,
                )
                raise self._map_exception(exc, duration_ms=total_duration_ms) from exc
