from personal_ai.core.logger import get_logger
from personal_ai.llm.client import LLMClient
from personal_ai.models.chat import ChatRequest, ChatResponse

logger = get_logger(__name__)


class ChatService:
    """Business logic service for managing chat processing.

    Depends exclusively on the LLMClient abstraction for LLM interactions.
    Does NOT depend on LiteLLM or any concrete provider implementation.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize ChatService with an abstract LLMClient instance.

        Args:
            llm_client: Abstract LLM client implementation.
        """
        self._llm_client = llm_client

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process an incoming chat request and return formatted ChatResponse.

        Args:
            request: Inbound chat request payload.

        Returns:
            ChatResponse: Processed chat response containing completion content and execution metadata.
        """
        logger.info("Processing chat request")

        llm_response = await self._llm_client.generate_response(
            prompt=request.message,
            system_prompt=request.system_prompt,
        )

        return ChatResponse(
            response=llm_response.content,
            provider=llm_response.provider,
            model=llm_response.model,
            latency_ms=llm_response.latency_ms,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens,
        )
