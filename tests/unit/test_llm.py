import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from personal_ai.config.settings import Settings
from personal_ai.llm import (
    LLMAuthenticationException,
    LLMClient,
    LLMConnectionException,
    LLMException,
    LLMMessage,
    LLMProvider,
    LLMRateLimitException,
    LLMResponse,
    LLMStreamChunk,
    get_llm_client,
)
from personal_ai.llm.litellm_client import LiteLLMClient


def test_abstract_llm_client_cannot_be_instantiated() -> None:
    """Verify that abstract LLMClient interface cannot be directly instantiated."""
    with pytest.raises(TypeError):
        LLMClient()  # type: ignore[abstract]


def test_factory_returns_llm_client() -> None:
    """Verify that get_llm_client returns an instance implementing LLMClient."""
    client = get_llm_client()
    assert isinstance(client, LLMClient)


def test_provider_enum_constants() -> None:
    """Verify provider enum constants."""
    assert LLMProvider.OPENAI == "openai"
    assert LLMProvider.ANTHROPIC == "anthropic"
    assert LLMProvider.GEMINI == "gemini"


def test_litellm_model_formatting() -> None:
    """Verify LiteLLM model string construction across providers."""
    client_openai = LiteLLMClient(provider="openai", model="gpt-4o")
    assert client_openai._format_model_name() == "openai/gpt-4o"

    client_anthropic = LiteLLMClient(
        provider="anthropic", model="claude-3-5-sonnet-20241022"
    )
    assert (
        client_anthropic._format_model_name()
        == "anthropic/claude-3-5-sonnet-20241022"
    )

    client_preformatted = LiteLLMClient(
        provider="openrouter", model="anthropic/claude-3-5-sonnet"
    )
    assert client_preformatted._format_model_name() == "anthropic/claude-3-5-sonnet"


@pytest.mark.asyncio
async def test_generate_response_returns_llm_response() -> None:
    """Verify successful LLM response generation returning structured LLMResponse."""
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 30

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Mocked response text"))
    ]
    mock_response.usage = mock_usage

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response

        client = LiteLLMClient(provider="openai", model="gpt-4o-mini")
        messages = [
            LLMMessage(role="system", content="You are a helpful AI"),
            LLMMessage(role="user", content="Hello"),
        ]
        result = await client.generate_response(messages=messages)

        assert isinstance(result, LLMResponse)
        assert result.content == "Mocked response text"
        assert result.provider == "openai"
        assert result.model == "gpt-4o-mini"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30
        assert result.latency_ms >= 0.0

        mock_acompletion.assert_called_once()
        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o-mini"
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "You are a helpful AI"},
            {"role": "user", "content": "Hello"},
        ]


@pytest.mark.asyncio
async def test_stream_response_yields_llm_stream_chunks() -> None:
    """Verify stream_response yields structured LLMStreamChunk objects."""

    class MockAsyncStream:
        def __init__(self, chunks):
            self._chunks = chunks

        def __aiter__(self):
            self._iter = iter(self._chunks)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="Hello"), finish_reason=None)]
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content=" world"), finish_reason="stop")]
    chunk2.usage = None

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = MockAsyncStream([chunk1, chunk2])

        client = LiteLLMClient(provider="openai", model="gpt-4o-mini")
        messages = [LLMMessage(role="user", content="Hello")]

        stream_gen = client.stream_response(messages=messages)
        chunks = [c async for c in stream_gen]

        assert len(chunks) == 2
        assert isinstance(chunks[0], LLMStreamChunk)
        assert chunks[0].content == "Hello"
        assert chunks[0].finish_reason is None
        assert chunks[1].content == " world"
        assert chunks[1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_exception_mapping_authentication() -> None:
    """Verify LiteLLM AuthenticationError is mapped to LLMAuthenticationException with sanitized message."""
    import litellm

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = litellm.exceptions.AuthenticationError(
            message="Invalid API Key secret_key_12345",
            response=MagicMock(),
            llm_provider="openai",
            model="gpt-4o-mini",
        )

        client = LiteLLMClient(provider="openai", model="gpt-4o-mini")
        with pytest.raises(LLMAuthenticationException) as exc_info:
            await client.generate_response([LLMMessage(role="user", content="Hello")])

        assert exc_info.value.status_code == 401
        assert "secret_key_12345" not in exc_info.value.message


@pytest.mark.asyncio
async def test_exception_mapping_rate_limit() -> None:
    """Verify LiteLLM RateLimitError is mapped to LLMRateLimitException."""
    import litellm

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = litellm.exceptions.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(),
            llm_provider="openai",
            model="gpt-4o-mini",
        )

        client = LiteLLMClient(provider="openai", model="gpt-4o-mini")
        with pytest.raises(LLMRateLimitException) as exc_info:
            await client.generate_response([LLMMessage(role="user", content="Hello")])

        assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_exception_mapping_connection() -> None:
    """Verify LiteLLM APIConnectionError is mapped to LLMConnectionException."""
    import litellm

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = litellm.exceptions.APIConnectionError(
            message="Connection failed", llm_provider="openai", model="gpt-4o-mini"
        )

        client = LiteLLMClient(provider="openai", model="gpt-4o-mini")
        with pytest.raises(LLMConnectionException) as exc_info:
            await client.generate_response([LLMMessage(role="user", content="Hello")])

        assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_acceptance_zero_code_change_provider_switch() -> None:
    """Acceptance test: Changing LLM_PROVIDER or LLM_MODEL in settings requires zero code changes."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
    mock_response.usage = None
    messages = [LLMMessage(role="user", content="Test prompt")]

    # Test Provider 1: OpenAI
    settings_openai = Settings(llm_provider="openai", llm_model="gpt-4o")
    client_1 = LiteLLMClient(settings=settings_openai)
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_1:
        mock_1.return_value = mock_response
        res1 = await client_1.generate_response(messages)
        assert res1.provider == "openai"
        assert mock_1.call_args.kwargs["model"] == "openai/gpt-4o"

    # Test Provider 2: Anthropic
    settings_anthropic = Settings(
        llm_provider="anthropic", llm_model="claude-3-5-sonnet-20241022"
    )
    client_2 = LiteLLMClient(settings=settings_anthropic)
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_2:
        mock_2.return_value = mock_response
        res2 = await client_2.generate_response(messages)
        assert res2.provider == "anthropic"
        assert mock_2.call_args.kwargs["model"] == "anthropic/claude-3-5-sonnet-20241022"

    # Test Provider 3: Ollama
    settings_ollama = Settings(llm_provider="ollama", llm_model="llama3")
    client_3 = LiteLLMClient(settings=settings_ollama)
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_3:
        mock_3.return_value = mock_response
        res3 = await client_3.generate_response(messages)
        assert res3.provider == "ollama"
        assert mock_3.call_args.kwargs["model"] == "ollama/llama3"


# ==============================================================================
# PR #15 Resilience Tests: Timeouts, Retries, and Exception Mapping
# ==============================================================================

class MockAsyncStream:
    """Helper to mock an asynchronous stream of LiteLLM chunks."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._iter = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


@pytest.mark.asyncio
async def test_stream_response_503_retry_success() -> None:
    """Verify transient 503 ServiceUnavailableError before first token is retried and succeeds."""
    import litellm

    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content="Retried success"), finish_reason="stop")]
    chunk.usage = None

    mock_stream = MockAsyncStream([chunk])

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        # Attempt 1 raises 503, Attempt 2 returns valid stream
        mock_acompletion.side_effect = [
            litellm.exceptions.ServiceUnavailableError(
                message="This model is currently experiencing high demand",
                response=MagicMock(),
                llm_provider="gemini",
                model="gemini-3.6-flash",
            ),
            mock_stream,
        ]

        settings = Settings(
            llm_provider="gemini",
            llm_model="gemini-3.6-flash",
            llm_max_retries=2,
            llm_retry_initial_delay=0.01,  # fast backoff for test
        )
        client = LiteLLMClient(settings=settings)
        messages = [LLMMessage(role="user", content="Hello")]

        stream_gen = client.stream_response(messages=messages)
        chunks = [c async for c in stream_gen]

        assert len(chunks) == 1
        assert chunks[0].content == "Retried success"
        assert mock_acompletion.call_count == 2


@pytest.mark.asyncio
async def test_stream_response_repeated_503_eventually_fails() -> None:
    """Verify persistent 503 eventually fails after bounded retries and raises LLMConnectionException."""
    import litellm

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = litellm.exceptions.ServiceUnavailableError(
            message="503 This model is currently experiencing high demand",
            response=MagicMock(),
            llm_provider="gemini",
            model="gemini-3.6-flash",
        )

        settings = Settings(
            llm_provider="gemini",
            llm_model="gemini-3.6-flash",
            llm_max_retries=2,
            llm_retry_initial_delay=0.01,
        )
        client = LiteLLMClient(settings=settings)
        messages = [LLMMessage(role="user", content="Hello")]

        with pytest.raises(LLMConnectionException) as exc_info:
            stream_gen = client.stream_response(messages=messages)
            _ = [c async for c in stream_gen]

        assert exc_info.value.status_code == 503
        assert "AI service is temporarily unavailable" in exc_info.value.message
        # Attempt 1 + 2 retries = 3 total attempts
        assert mock_acompletion.call_count == 3


@pytest.mark.asyncio
async def test_stream_response_timeout_bounded() -> None:
    """Verify stream start timeout triggers bounded failure and raises LLMConnectionException."""
    async def slow_stream(*args, **kwargs):
        await asyncio.sleep(0.5)
        return MagicMock()

    with patch("litellm.acompletion", side_effect=slow_stream):
        settings = Settings(
            llm_provider="gemini",
            llm_model="gemini-3.6-flash",
            llm_stream_start_timeout=0.05,  # short timeout for test
            llm_max_retries=1,
            llm_retry_initial_delay=0.01,
        )
        client = LiteLLMClient(settings=settings)
        messages = [LLMMessage(role="user", content="Hello")]

        start_time = time.perf_counter()
        with pytest.raises(LLMConnectionException) as exc_info:
            stream_gen = client.stream_response(messages=messages)
            _ = [c async for c in stream_gen]
        elapsed = time.perf_counter() - start_time

        assert elapsed < 0.3  # Bounded execution, fails fast
        assert "AI service is temporarily unavailable" in exc_info.value.message


@pytest.mark.asyncio
async def test_generate_response_503_retry_success() -> None:
    """Verify synchronous generate_response retries transient 503 and returns LLMResponse."""
    import litellm

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Success after retry"))]
    mock_response.usage = None

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = [
            litellm.exceptions.ServiceUnavailableError(
                message="503 Service Unavailable",
                response=MagicMock(),
                llm_provider="gemini",
                model="gemini-3.6-flash",
            ),
            mock_response,
        ]

        settings = Settings(
            llm_provider="gemini",
            llm_model="gemini-3.6-flash",
            llm_max_retries=2,
            llm_retry_initial_delay=0.01,
        )
        client = LiteLLMClient(settings=settings)
        res = await client.generate_response([LLMMessage(role="user", content="Hello")])

        assert res.content == "Success after retry"
        assert mock_acompletion.call_count == 2


@pytest.mark.asyncio
async def test_non_transient_error_does_not_retry() -> None:
    """Verify non-transient error (e.g. 401 AuthenticationError) fails immediately without retry."""
    import litellm

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = litellm.exceptions.AuthenticationError(
            message="Invalid API Key",
            response=MagicMock(),
            llm_provider="openai",
            model="gpt-4o-mini",
        )

        settings = Settings(llm_max_retries=2)
        client = LiteLLMClient(settings=settings)

        with pytest.raises(LLMAuthenticationException):
            await client.generate_response([LLMMessage(role="user", content="Hello")])

        # Exactly 1 attempt, no retries wasted on 401
        assert mock_acompletion.call_count == 1


@pytest.mark.asyncio
async def test_clean_exception_mapping_vertex_beta() -> None:
    """Verify custom Vertex_ai_betaException / MidStreamFallbackError is cleanly mapped to domain exception."""
    import litellm

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = litellm.exceptions.MidStreamFallbackError(
            message="Vertex_ai_betaException: 503 Model overloaded",
            llm_provider="gemini",
            model="gemini-3.6-flash",
        )

        settings = Settings(llm_max_retries=0)
        client = LiteLLMClient(settings=settings)

        with pytest.raises(LLMConnectionException) as exc_info:
            stream_gen = client.stream_response([LLMMessage(role="user", content="Hello")])
            _ = [c async for c in stream_gen]

        assert exc_info.value.status_code == 503
        assert "AI service is temporarily unavailable" in exc_info.value.message
        # Ensure internal vertex beta exception details are sanitized
        assert "Vertex_ai_betaException" not in exc_info.value.message


@pytest.mark.asyncio
async def test_stream_response_mid_stream_chunk_timeout() -> None:
    """Verify stream chunk timeout terminates mid-stream hang after yielding initial chunks."""
    class MockHangingStream:
        def __init__(self):
            self.step = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.step += 1
            if self.step == 1:
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content="First"), finish_reason=None)]
                chunk.usage = None
                return chunk
            elif self.step == 2:
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content=" Second"), finish_reason=None)]
                chunk.usage = None
                return chunk
            else:
                # 3rd chunk hangs indefinitely
                await asyncio.sleep(2.0)
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content=" Third"), finish_reason="stop")]
                chunk.usage = None
                return chunk

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = MockHangingStream()

        settings = Settings(
            llm_provider="gemini",
            llm_model="gemini-3.6-flash",
            llm_stream_start_timeout=1.0,
            llm_stream_chunk_timeout=0.05,  # 50ms chunk timeout
        )
        client = LiteLLMClient(settings=settings)
        messages = [LLMMessage(role="user", content="Hello")]

        stream_gen = client.stream_response(messages=messages)
        yielded_chunks = []

        start_time = time.perf_counter()
        with pytest.raises(LLMConnectionException) as exc_info:
            async for c in stream_gen:
                yielded_chunks.append(c.content)

        elapsed = time.perf_counter() - start_time

        # Initial 2 chunks were received
        assert yielded_chunks == ["First", " Second"]
        # Timed out quickly on 3rd chunk (< 0.2s instead of hanging for 2.0s)
        assert elapsed < 0.2
        assert "AI service is temporarily unavailable" in exc_info.value.message


@pytest.mark.asyncio
async def test_midstream_fallback_error_retry_stream_start() -> None:
    """Verify MidStreamFallbackError is treated as transient and retried during stream start."""
    import litellm

    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content="Success after fallback"), finish_reason="stop")]
    chunk.usage = None

    mock_stream = MockAsyncStream([chunk])

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = [
            litellm.exceptions.MidStreamFallbackError(
                message="MidStreamFallbackError: Vertex AI 503 high demand",
                llm_provider="gemini",
                model="gemini-3.6-flash",
            ),
            mock_stream,
        ]

        settings = Settings(
            llm_provider="gemini",
            llm_model="gemini-3.6-flash",
            llm_max_retries=2,
            llm_retry_initial_delay=0.01,
        )
        client = LiteLLMClient(settings=settings)
        messages = [LLMMessage(role="user", content="Hello")]

        stream_gen = client.stream_response(messages=messages)
        chunks = [c async for c in stream_gen]

        assert len(chunks) == 1
        assert chunks[0].content == "Success after fallback"
        assert mock_acompletion.call_count == 2

