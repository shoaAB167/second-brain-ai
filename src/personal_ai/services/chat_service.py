from typing import List

from personal_ai.core.exceptions import AppException
from personal_ai.core.logger import get_logger
from personal_ai.db.repositories.base import ConversationRepository
from personal_ai.llm.client import LLMClient
from personal_ai.llm.models import LLMMessage
from personal_ai.models.chat import ChatRequest, ChatResponse

logger = get_logger(__name__)


class ChatService:
    """Business logic service for managing chat processing and conversation memory.

    Depends exclusively on LLMClient and ConversationRepository abstract interfaces.
    Does NOT depend on LiteLLM, SQLAlchemy, or concrete database implementations.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        conversation_repo: ConversationRepository,
    ) -> None:
        """Initialize ChatService with abstract dependencies.

        Args:
            llm_client: Abstract LLM client interface.
            conversation_repo: Abstract conversation repository interface.
        """
        self._llm_client = llm_client
        self._conversation_repo = conversation_repo

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process an incoming chat request, maintaining conversation context.

        Transaction Execution Strategy:
        -------------------------------
        1. Resolve existing conversation or create a new conversation thread.
        2. Retrieve stored conversation messages in chronological order.
        3. Convert stored messages to domain LLMMessage format and append new user prompt.
        4. PERSIST USER MESSAGE: The user's prompt is committed to history first.
        5. CALL LLM: Invoke LLMClient.generate_response(messages).
        6. PERSIST ASSISTANT RESPONSE: If LLM succeeds, commit assistant response to history.
           FALLBACK ON FAILURE: If LLM fails (raises LLMException), the user message remains
           recorded in history as an attempted message, but NO assistant response is created.

        Args:
            request: Inbound chat request payload.

        Returns:
            ChatResponse: Processed chat response containing completion content,
                         conversation ID, and execution metadata.

        Raises:
            AppException: If conversation ID is provided but not found (404).
        """
        # 1. Resolve or create conversation thread
        if request.conversation_id:
            conversation = await self._conversation_repo.get_conversation(
                request.conversation_id
            )
            if not conversation:
                logger.warning(
                    "Requested conversation not found [conversation_id=%s]",
                    request.conversation_id,
                )
                raise AppException(
                    message=f"Conversation '{request.conversation_id}' not found.",
                    status_code=404,
                )
        else:
            conversation = await self._conversation_repo.create_conversation()
            logger.info(
                "Created new conversation [conversation_id=%s]",
                conversation.id,
            )

        # 2. Retrieve previous conversation messages in chronological order (oldest -> newest)
        stored_messages = await self._conversation_repo.get_conversation_messages(
            conversation.id
        )

        # 3. Convert stored messages to domain LLMMessage representation
        history: List[LLMMessage] = []

        if request.system_prompt:
            history.append(
                LLMMessage(role="system", content=request.system_prompt)
            )

        for msg in stored_messages:
            history.append(
                LLMMessage(
                    role=msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                    content=msg.content,
                )
            )

        # Append current user prompt message to history payload
        history.append(LLMMessage(role="user", content=request.message))

        # 4. Persist user message to conversation history before calling LLM
        await self._conversation_repo.add_message(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
        )

        # 5. Execute LLM completion request with complete conversation context
        llm_response = await self._llm_client.generate_response(messages=history)

        # 6. Persist assistant response to conversation history after successful LLM call
        await self._conversation_repo.add_message(
            conversation_id=conversation.id,
            role="assistant",
            content=llm_response.content,
        )

        # 7. Return formatted ChatResponse containing conversation_id
        return ChatResponse(
            conversation_id=conversation.id,
            response=llm_response.content,
            provider=llm_response.provider,
            model=llm_response.model,
            latency_ms=llm_response.latency_ms,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens,
        )
