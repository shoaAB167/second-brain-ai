from datetime import datetime
from typing import List, Optional
import uuid

from pydantic import BaseModel, Field


class MemorySearchResultItem(BaseModel):
    """Schema representing an individual retrieved memory search result."""

    experience_id: uuid.UUID = Field(
        ...,
        alias="experienceId",
        description="UUID of the matched Experience record.",
    )
    type: Optional[str] = Field(
        None,
        description="Extracted experience classification type (e.g., GOAL, FACT, PREFERENCE).",
    )
    content: str = Field(
        ...,
        description="Structured content of the Experience.",
    )
    domain: Optional[str] = Field(
        None,
        description="Knowledge domain (e.g. career, personal, health).",
    )
    status: str = Field(
        ...,
        description="Processing status of the Experience record.",
    )
    similarity: float = Field(
        ...,
        description="Cosine similarity score between query and experience embedding (in [-1.0, 1.0]).",
    )
    source_message_id: Optional[uuid.UUID] = Field(
        None,
        alias="sourceMessageId",
        description="UUID of the source conversation message if originated from chat.",
    )
    created_at: datetime = Field(
        ...,
        alias="createdAt",
        description="Creation timestamp of the Experience record.",
    )

    model_config = {
        "populate_by_name": True,
    }


class MemorySearchResponse(BaseModel):
    """Response payload for GET /api/v1/memories/search."""

    query: str = Field(..., description="The executed natural language search query.")
    count: int = Field(..., description="Total matching memories returned.")
    results: List[MemorySearchResultItem] = Field(
        default_factory=list,
        description="Ranked list of memory search results ordered by similarity.",
    )

    model_config = {
        "populate_by_name": True,
    }
