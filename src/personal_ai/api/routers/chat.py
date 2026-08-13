from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.db.repositories import (
    ConversationRepository,
    SQLAlchemyConversationRepository,
)
from personal_ai.db.session import get_db_session
from personal_ai.llm import LLMClient, get_llm_client
from personal_ai.models.chat import ChatRequest, ChatResponse
from personal_ai.services.chat_service import ChatService

router = APIRouter()


def get_conversation_repository(
    session: AsyncSession = Depends(get_db_session),
) -> ConversationRepository:
    """Dependency provider constructing concrete SQLAlchemyConversationRepository as ConversationRepository abstraction."""
    return SQLAlchemyConversationRepository(session=session)


def get_chat_service(
    llm_client: LLMClient = Depends(get_llm_client),
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
) -> ChatService:
    """Dependency provider for ChatService, injecting repository and LLM abstractions."""
    return ChatService(llm_client=llm_client, conversation_repo=conversation_repo)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send Chat Message",
    description="Processes a prompt message with conversation history synchronously.",
)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Execute synchronous chat completion request through ChatService."""
    return await chat_service.process_chat(request)


@router.post(
    "/chat/stream",
    summary="Send Chat Message (Streaming)",
    description="Streams chat completions using Server-Sent Events (SSE) in real time.",
)
async def chat_stream(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """Execute streaming chat completion request returning Server-Sent Events (SSE)."""
    return StreamingResponse(
        chat_service.process_chat_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
