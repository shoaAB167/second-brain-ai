from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from personal_ai.config.settings import Settings
from personal_ai.llm import (
    LLMAuthenticationException,
    LLMClient,
    LLMConnectionException,
    LLMException,
    LLMModel,
    LLMProvider,
    LLMRateLimitException,
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


def test_model_constants_and_enums() -> None:
    """Verify provider and model enum constants."""
    assert LLMProvider.OPENAI == "openai"
    assert LLMProvider.ANTHROPIC == "anthropic"
    assert LLMProvider.GEMINI == "gemini"
    assert LLMModel.GPT_4O_MINI == "gpt-4o-mini"
    assert LLMModel.CLAUDE_3_5_SONNET == "claude-3-5-sonnet-20241022"


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
async def test_generate_response_success() -> None:
    """Verify successful LLM response generation with system prompt and message formatting."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Mocked response text"))
    ]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response

        client = LiteLLMClient(provider="openai", model="gpt-4o-mini")
        response = await client.generate_response(
            prompt="Hello", system_prompt="You are a helpful AI"
        )

        assert response == "Mocked response text"
        mock_acompletion.assert_called_once()

        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4o-mini"
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "You are a helpful AI"},
            {"role": "user", "content": "Hello"},
        ]


@pytest.mark.asyncio
async def test_exception_mapping_authentication() -> None:
    """Verify LiteLLM AuthenticationError is mapped to LLMAuthenticationException."""
    import litellm

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = litellm.exceptions.AuthenticationError(
            message="Invalid API Key", response=MagicMock(), llm_provider="openai", model="gpt-4o-mini"
        )

        client = LiteLLMClient(provider="openai", model="gpt-4o-mini")
        with pytest.raises(LLMAuthenticationException) as exc_info:
            await client.generate_response("Hello")

        assert exc_info.value.status_code == 401
        assert "authentication failed" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_exception_mapping_rate_limit() -> None:
    """Verify LiteLLM RateLimitError is mapped to LLMRateLimitException."""
    import litellm

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = litellm.exceptions.RateLimitError(
            message="Rate limit exceeded", response=MagicMock(), llm_provider="openai", model="gpt-4o-mini"
        )

        client = LiteLLMClient(provider="openai", model="gpt-4o-mini")
        with pytest.raises(LLMRateLimitException) as exc_info:
            await client.generate_response("Hello")

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
            await client.generate_response("Hello")

        assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_health_check_success_and_failure() -> None:
    """Verify health_check returns True on success and False on error."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="pong"))]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response
        client = LiteLLMClient(provider="openai", model="gpt-4o-mini")
        assert await client.health_check() is True

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = Exception("Service unavailable")
        client = LiteLLMClient(provider="openai", model="gpt-4o-mini")
        assert await client.health_check() is False


@pytest.mark.asyncio
async def test_acceptance_zero_code_change_provider_switch() -> None:
    """Acceptance test: Changing LLM_PROVIDER or LLM_MODEL in settings requires zero code changes."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]

    # Test Provider 1: OpenAI
    settings_openai = Settings(llm_provider="openai", llm_model="gpt-4o")
    client_1 = LiteLLMClient(settings=settings_openai)
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_1:
        mock_1.return_value = mock_response
        await client_1.generate_response("Test prompt")
        assert mock_1.call_args.kwargs["model"] == "openai/gpt-4o"

    # Test Provider 2: Anthropic
    settings_anthropic = Settings(
        llm_provider="anthropic", llm_model="claude-3-5-sonnet-20241022"
    )
    client_2 = LiteLLMClient(settings=settings_anthropic)
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_2:
        mock_2.return_value = mock_response
        await client_2.generate_response("Test prompt")
        assert mock_2.call_args.kwargs["model"] == "anthropic/claude-3-5-sonnet-20241022"

    # Test Provider 3: Ollama
    settings_ollama = Settings(llm_provider="ollama", llm_model="llama3")
    client_3 = LiteLLMClient(settings=settings_ollama)
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_3:
        mock_3.return_value = mock_response
        await client_3.generate_response("Test prompt")
        assert mock_3.call_args.kwargs["model"] == "ollama/llama3"
