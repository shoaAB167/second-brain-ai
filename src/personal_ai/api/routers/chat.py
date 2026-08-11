from fastapi import APIRouter, Depends

from personal_ai.llm import LLMClient, get_llm_client
from personal_ai.models.chat import ChatRequest, ChatResponse
from personal_ai.services.chat_service import ChatService

router = APIRouter()


def get_chat_service(llm_client: LLMClient = Depends(get_llm_client)) -> ChatService:
    """Dependency provider for ChatService.

    Injects the abstract LLMClient instance into ChatService.
    """
    return ChatService(llm_client=llm_client)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send Chat Message",
    description="Processes a prompt message using the configured LLM backend.",
)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Execute chat completion request through the ChatService."""
    return await chat_service.process_chat(request)
