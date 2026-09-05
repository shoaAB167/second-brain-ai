from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field

from personal_ai.application.memory import PersonalContextRetrievalService
from personal_ai.core.logger import get_logger
from personal_ai.domain.tool import BaseTool, ToolExecutionContext, ToolPermission

logger = get_logger(__name__)


class SearchPersonalMemoryInput(BaseModel):
    """Declarative input schema for searching personal memory.
    
    Rejects extra arguments (such as attempted user_id injection) via strict validation.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The search query text to find relevant memories and past experiences.",
    )
    limit: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of memories to return (1-10, default is 5).",
    )


class SearchPersonalMemoryTool(BaseTool):
    """Controlled, read-only capability for searching authenticated user personal memories.
    
    Delegates strictly to PersonalContextRetrievalService using the application-provided
    ToolExecutionContext for user isolation. The LLM cannot specify or override user_id.
    """

    name = "search_personal_memory"
    description = (
        "Search the authenticated user's long-term personal memory and experiences by "
        "semantic similarity and dimensions."
    )
    permission = ToolPermission.READ_ONLY
    input_schema = SearchPersonalMemoryInput

    def __init__(self, retrieval_service: PersonalContextRetrievalService) -> None:
        """Initialize tool with injected PersonalContextRetrievalService."""
        self._retrieval_service = retrieval_service

    async def _run(
        self,
        query: str,
        limit: Optional[int] = 5,
        context: Optional[ToolExecutionContext] = None,
    ) -> Dict[str, Any]:
        """Execute memory retrieval bounded to authenticated user context.

        Args:
            query: Search query text.
            limit: Maximum items to retrieve (bounded 1-10).
            context: Application-supplied execution context holding the authenticated user_id.

        Returns:
            Structured dictionary containing bounded list of retrieved memories.

        Raises:
            ValueError: If execution context or authenticated user_id is missing.
        """
        if not context or not context.user_id:
            logger.error("SearchPersonalMemoryTool invoked without authenticated user context.")
            raise ValueError("Authenticated user context is required to search personal memory.")

        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Search query cannot be empty.")

        effective_limit = max(1, min(limit or 5, 10))

        logger.info(
            "Executing SearchPersonalMemoryTool [user_id=%s, query_len=%d, limit=%d]",
            context.user_id,
            len(clean_query),
            effective_limit,
        )

        personal_context = await self._retrieval_service.retrieve_context(
            user_id=context.user_id,
            query=clean_query,
            final_limit=effective_limit,
        )

        memories: List[Dict[str, Any]] = []
        for item in personal_context.items:
            memories.append(
                {
                    "content": item.content,
                    "type": item.type or "EXPERIENCE",
                    "domain": item.domain or "general",
                    "importance": item.importance,
                    "emotion": (
                        item.emotional_context.get("emotion")
                        if item.emotional_context
                        else None
                    ),
                    "evidence_level": item.evidence_level or "EXTRACTED",
                }
            )

        return {
            "memories": memories,
            "count": len(memories),
        }
