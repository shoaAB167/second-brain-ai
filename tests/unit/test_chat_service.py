from unittest.mock import AsyncMock, MagicMock

import pytest

from personal_ai.llm.client import LLMClient
from personal_ai.llm.models import LLMResponse
from personal_ai.models.chat import ChatRequest, ChatResponse
from personal_ai.services.chat_service import ChatService


@pytest.mark.asyncio
async def test_chat_service_process_chat() -> None:
    """Verify ChatService processes request using LLMClient abstraction."""
    mock_llm_client = MagicMock(spec=LLMClient)
    mock_llm_client.generate_response = AsyncMock(
        return_value=LLMResponse(
            content="Service answer",
            provider="openai",
            model="gpt-4o-mini",
            latency_ms=120.5,
            prompt_tokens=15,
            completion_tokens=25,
            total_tokens=40,
        )
    )

    chat_service = ChatService(llm_client=mock_llm_client)
    request = ChatRequest(message="Hello AI", system_prompt="Be concise")

    response = await chat_service.process_chat(request)

    assert isinstance(response, ChatResponse)
    assert response.response == "Service answer"
    assert response.provider == "openai"
    assert response.model == "gpt-4o-mini"
    assert response.latency_ms == 120.5
    assert response.prompt_tokens == 15
    assert response.completion_tokens == 25
    assert response.total_tokens == 40

    mock_llm_client.generate_response.assert_called_once_with(
        prompt="Hello AI",
        system_prompt="Be concise",
    )
