from typing import List

from personal_ai.core.exceptions import AppException
from personal_ai.core.logger import get_logger
from personal_ai.db.repositories.conversation_repository import ConversationRepository
from personal_ai.llm.client import LLMClient
from personal_ai.llm.models import LLMMessage
from personal_ai.models.chat import ChatRequest, ChatResponse

logger = get_logger(__name__)


class ChatService:
    """Business logic service for managing chat processing and conversation memory.

    Depends exclusively on LLMClient and ConversationRepository abstractions.
    Does NOT depend on LiteLLM or any concrete provider implementation.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        conversation_repo: ConversationRepository,
    ) -> None:
        """Initialize ChatService with abstract dependencies.

        Args:
            llm_client: Abstract LLM client interface.
            conversation_repo: Conversation repository for persistence.
        """
        self._llm_client = llm_client
        self._conversation_repo = conversation_repo

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process an incoming chat request, maintaining conversation context.

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

        # 2. Retrieve previous conversation messages in chronological order
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
                LLMMessage(role=msg.role.value if hasattr(msg.role, "value") else str(msg.role), content=msg.content)
            )

        # Append current user prompt message to history
        history.append(LLMMessage(role="user", content=request.message))

        # 4. Persist user message to conversation history
        await self._conversation_repo.add_message(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
        )

        # 5. Execute LLM completion request with complete conversation context
        llm_response = await self._llm_client.generate_response(messages=history)

        # 6. Persist assistant response to conversation history
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
