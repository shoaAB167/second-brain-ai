from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock
import uuid
import pytest

from personal_ai.application.memory import (
    MemoryContextBuilder,
    MemoryRetrievalService,
    MemorySearchResult,
)
from personal_ai.core.exceptions import AppException
from personal_ai.db.models import Conversation, Message
from personal_ai.db.repositories.base import ConversationRepository
from personal_ai.llm.client import LLMClient
from personal_ai.llm.models import LLMMessage, LLMResponse
from personal_ai.models.chat import ChatRequest
from personal_ai.services.chat_service import ChatService


class MockConversationRepository(ConversationRepository):
    """In-memory conversation repository for testing ChatService."""

    def __init__(self) -> None:
        self.conversations: List[Conversation] = []
        self.messages: List[Message] = []

    async def create_conversation(self, title: Optional[str] = None, user_id: Optional[uuid.UUID] = None) -> Conversation:
        conv = Conversation(id=uuid.uuid4(), user_id=user_id)
        self.conversations.append(conv)
        return conv

    async def get_conversation(self, conversation_id: uuid.UUID, user_id: Optional[uuid.UUID] = None) -> Optional[Conversation]:
        for c in self.conversations:
            if c.id == conversation_id:
                if user_id and c.user_id and c.user_id != user_id:
                    return None
                return c
        return None

    async def list_conversations(self, user_id: Optional[uuid.UUID] = None, limit: int = 50, offset: int = 0) -> List[Conversation]:
        return [c for c in self.conversations if user_id is None or c.user_id == user_id]

    async def delete_conversation(self, conversation_id: uuid.UUID, user_id: Optional[uuid.UUID] = None) -> bool:
        for i, c in enumerate(self.conversations):
            if c.id == conversation_id:
                self.conversations.pop(i)
                return True
        return False

    async def add_message(self, conversation_id: uuid.UUID, role: str, content: str) -> Message:
        msg = Message(id=uuid.uuid4(), conversation_id=conversation_id, role=role, content=content)
        self.messages.append(msg)
        return msg

    async def get_conversation_messages(self, conversation_id: uuid.UUID, limit: Optional[int] = None) -> List[Message]:
        msgs = [m for m in self.messages if m.conversation_id == conversation_id]
        return msgs[:limit] if limit else msgs


@pytest.mark.asyncio
async def test_17a_chat_with_relevant_memories_injects_context() -> None:
    """Requirement 17A: When memory retrieval returns memories, LLM receives memory context block."""
    user_id = uuid.uuid4()
    mock_llm = MagicMock(spec=LLMClient)
    captured_messages: List[LLMMessage] = []

    async def fake_generate(messages: List[LLMMessage], **kwargs) -> LLMResponse:
        nonlocal captured_messages
        captured_messages = list(messages)
        return LLMResponse(
            content="To reach 30 LPA, focus on system design and distributed architectures.",
            provider="gemini",
            model="gemini-3.5-flash",
            latency_ms=15.0,
        )

    mock_llm.generate_response = AsyncMock(side_effect=fake_generate)
    conv_repo = MockConversationRepository()

    retrieved_memory = MemorySearchResult(
        experience_id=uuid.uuid4(),
        type="GOAL",
        domain="career",
        content="Reach a salary of 30 LPA",
        status="RECEIVED",
        similarity=0.94,
        source_message_id=uuid.uuid4(),
        created_at=datetime.now(timezone.utc),
    )

    mock_retrieval = MagicMock(spec=MemoryRetrievalService)
    mock_retrieval.search = AsyncMock(return_value=[retrieved_memory])

    service = ChatService(
        llm_client=mock_llm,
        conversation_repo=conv_repo,
        retrieval_service=mock_retrieval,
    )

    req = ChatRequest(message="What should I focus on to reach my career goal?")
    res = await service.process_chat(req, user_id=user_id)

    assert "30 LPA" in res.response
    # Verify retrieval service was called with user_id and query
    mock_retrieval.search.assert_awaited_once()

    # Verify LLM received memory context in system prompt
    assert len(captured_messages) >= 2
    system_msg = captured_messages[0]
    assert system_msg.role == "system"
    assert "<user_memory>" in system_msg.content
    assert "Reach a salary of 30 LPA" in system_msg.content
    assert "Type: GOAL" in system_msg.content


@pytest.mark.asyncio
async def test_17b_chat_with_no_memories_proceeds_unaugmented() -> None:
    """Requirement 17B: When memory retrieval returns empty list [], LLM receives no memory context."""
    user_id = uuid.uuid4()
    mock_llm = MagicMock(spec=LLMClient)
    captured_messages: List[LLMMessage] = []

    async def fake_generate(messages: List[LLMMessage], **kwargs) -> LLMResponse:
        nonlocal captured_messages
        captured_messages = list(messages)
        return LLMResponse(
            content="The capital of France is Paris.",
            provider="gemini",
            model="gemini-3.5-flash",
            latency_ms=10.0,
        )

    mock_llm.generate_response = AsyncMock(side_effect=fake_generate)
    conv_repo = MockConversationRepository()

    mock_retrieval = MagicMock(spec=MemoryRetrievalService)
    mock_retrieval.search = AsyncMock(return_value=[])

    service = ChatService(
        llm_client=mock_llm,
        conversation_repo=conv_repo,
        retrieval_service=mock_retrieval,
    )

    req = ChatRequest(message="What is the capital of France?")
    res = await service.process_chat(req, user_id=user_id)

    assert "Paris" in res.response
    # System message should NOT contain <user_memory>
    for msg in captured_messages:
        assert "<user_memory>" not in msg.content


@pytest.mark.asyncio
async def test_17c_memory_retrieval_failure_does_not_break_chat() -> None:
    """Requirement 17C & 3: If MemoryRetrievalService raises an exception, chat completes successfully."""
    user_id = uuid.uuid4()
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(
            content="Fallback answer",
            provider="gemini",
            model="gemini-3.5-flash",
            latency_ms=10.0,
        )
    )
    conv_repo = MockConversationRepository()

    mock_retrieval = MagicMock(spec=MemoryRetrievalService)
    mock_retrieval.search = AsyncMock(side_effect=RuntimeError("Vector search engine unavailable"))

    service = ChatService(
        llm_client=mock_llm,
        conversation_repo=conv_repo,
        retrieval_service=mock_retrieval,
    )

    req = ChatRequest(message="Tell me a joke.")
    res = await service.process_chat(req, user_id=user_id)

    assert res.response == "Fallback answer"


@pytest.mark.asyncio
async def test_17d_query_embedding_failure_does_not_break_chat() -> None:
    """Requirement 17D & 15: If embedding provider fails, chat continues without memory."""
    user_id = uuid.uuid4()
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(
            content="Normal answer without memories",
            provider="gemini",
            model="gemini-3.5-flash",
            latency_ms=10.0,
        )
    )
    conv_repo = MockConversationRepository()

    mock_retrieval = MagicMock(spec=MemoryRetrievalService)
    mock_retrieval.search = AsyncMock(side_effect=AppException("Failed to generate query embedding", status_code=502))

    service = ChatService(
        llm_client=mock_llm,
        conversation_repo=conv_repo,
        retrieval_service=mock_retrieval,
    )

    req = ChatRequest(message="Hello there!")
    res = await service.process_chat(req, user_id=user_id)

    assert res.response == "Normal answer without memories"


@pytest.mark.asyncio
async def test_17e_user_isolation_passes_correct_user_id() -> None:
    """Requirement 17E & 2: Chat passes current authenticated user_id to MemoryRetrievalService."""
    user_a = uuid.uuid4()
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content="OK", provider="gemini", model="gemini-3.5-flash", latency_ms=10.0)
    )
    conv_repo = MockConversationRepository()

    mock_retrieval = MagicMock(spec=MemoryRetrievalService)
    mock_retrieval.search = AsyncMock(return_value=[])

    service = ChatService(
        llm_client=mock_llm,
        conversation_repo=conv_repo,
        retrieval_service=mock_retrieval,
    )

    await service.process_chat(ChatRequest(message="Test query"), user_id=user_a)

    mock_retrieval.search.assert_awaited_once()
    assert mock_retrieval.search.call_args.kwargs["user_id"] == user_a


@pytest.mark.asyncio
async def test_17f_existing_conversation_history_and_message_preserved() -> None:
    """Requirement 17F & 17G: Short-term history and current message are preserved in chronological order."""
    user_id = uuid.uuid4()
    mock_llm = MagicMock(spec=LLMClient)
    captured_messages: List[LLMMessage] = []

    async def fake_generate(messages: List[LLMMessage], **kwargs) -> LLMResponse:
        nonlocal captured_messages
        captured_messages = list(messages)
        return LLMResponse(content="I remember!", provider="gemini", model="gemini-3.5-flash", latency_ms=10.0)

    mock_llm.generate_response = AsyncMock(side_effect=fake_generate)
    conv_repo = MockConversationRepository()

    # Pre-seed prior conversation
    conv = await conv_repo.create_conversation(user_id=user_id)
    await conv_repo.add_message(conversation_id=conv.id, role="user", content="My name is Shoaib.")
    await conv_repo.add_message(conversation_id=conv.id, role="assistant", content="Nice to meet you, Shoaib.")

    mock_retrieval = MagicMock(spec=MemoryRetrievalService)
    mock_retrieval.search = AsyncMock(return_value=[])

    service = ChatService(
        llm_client=mock_llm,
        conversation_repo=conv_repo,
        retrieval_service=mock_retrieval,
    )

    req = ChatRequest(conversation_id=conv.id, message="What is my name?")
    await service.process_chat(req, user_id=user_id)

    # Verify history order: prior user msg -> prior assistant msg -> new user msg
    user_msgs = [m for m in captured_messages if m.role == "user"]
    assistant_msgs = [m for m in captured_messages if m.role == "assistant"]

    assert len(user_msgs) == 2
    assert user_msgs[0].content == "My name is Shoaib."
    assert user_msgs[1].content == "What is my name?"
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].content == "Nice to meet you, Shoaib."


@pytest.mark.asyncio
async def test_17h_llm_called_exactly_once() -> None:
    """Requirement 17H: LLM client generate_response is called exactly once per chat turn."""
    user_id = uuid.uuid4()
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_response = AsyncMock(
        return_value=LLMResponse(content="Single turn response", provider="gemini", model="gemini-3.5-flash", latency_ms=10.0)
    )
    conv_repo = MockConversationRepository()

    mock_retrieval = MagicMock(spec=MemoryRetrievalService)
    mock_retrieval.search = AsyncMock(return_value=[])

    service = ChatService(
        llm_client=mock_llm,
        conversation_repo=conv_repo,
        retrieval_service=mock_retrieval,
    )

    await service.process_chat(ChatRequest(message="Single call test"), user_id=user_id)
    assert mock_llm.generate_response.call_count == 1
