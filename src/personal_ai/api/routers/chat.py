from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from personal_ai.db.repositories.conversation_repository import ConversationRepository
from personal_ai.db.session import get_db_session
from personal_ai.llm import LLMClient, get_llm_client
from personal_ai.models.chat import ChatRequest, ChatResponse
from personal_ai.services.chat_service import ChatService

router = APIRouter()


def get_conversation_repository(
    session: AsyncSession = Depends(get_db_session),
) -> ConversationRepository:
    """Dependency provider for ConversationRepository."""
    return ConversationRepository(session=session)


def get_chat_service(
    llm_client: LLMClient = Depends(get_llm_client),
    conversation_repo: ConversationRepository = Depends(get_conversation_repository),
) -> ChatService:
    """Dependency provider for ChatService."""
    return ChatService(llm_client=llm_client, conversation_repo=conversation_repo)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send Chat Message",
    description="Processes a prompt message with conversation history using the configured LLM backend.",
)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Execute chat completion request through ChatService."""
    return await chat_service.process_chat(request)
