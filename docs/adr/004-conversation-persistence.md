# ADR 004: Persistent Conversation Memory Architecture

## Status
Accepted

## Context
Second Brain AI requires a mechanism to store and retrieve multi-turn conversation context across user interactions. Without persistent memory, each request to the chat API would be isolated, losing all context from prior exchanges. 

As we lay the foundation for a future personal intelligence assistant, we need a reliable, structured persistence model for conversation history before introducing advanced cognitive capabilities such as semantic memory, vector search, or autonomous agents.

## Decision

We decide to implement relational conversation persistence in PostgreSQL using SQLAlchemy 2.x (asyncpg) and Alembic migrations, following strict repository abstraction principles.

### Key Architectural Choices

1. **Repository Abstraction (`ConversationRepository` ABC)**
   - **Rationale**: `ChatService` depends exclusively on the `ConversationRepository` Abstract Base Class interface, not on concrete SQLAlchemy or database session classes. The concrete `SQLAlchemyConversationRepository` is constructed by FastAPI's dependency injection layer.
   - **Dependency Graph**: `ChatService` ──► `ConversationRepository` (ABC) ◄── `SQLAlchemyConversationRepository` ──► `AsyncSession`.

2. **Entity Separation: `Conversation` and `Message`**
   - **Rationale**: Decoupling the conversation thread from individual messages enforces a 1-to-many relationship (`Conversation` 1 ──< `Message`). This allows independent message querying, pagination, indexing, soft-deletions, and metadata tracking per conversation.

3. **Last Activity Timestamp Tracking (`Conversation.updated_at`)**
   - **Rationale**: Every call to `add_message()` explicitly updates the parent `Conversation.updated_at` timestamp. This ensures `updated_at` reflects the true last activity timestamp of the conversation thread.

4. **Explicit Transaction Execution Strategy**
   - **Rationale**: When processing a chat request:
     1. Resolve or create `Conversation`.
     2. Retrieve stored messages in chronological order.
     3. **Persist user message**: Commit the user message to history *before* invoking the LLM.
     4. **Invoke LLM**: Call `LLMClient.generate_response(messages)`.
     5. **Persist assistant response**: Commit the assistant message to history if the LLM request succeeds.
     6. **Fallback on Failure**: If the LLM request fails (raises `LLMException`), the user message remains recorded as an attempted request, but no assistant message is created.

5. **Authentication & Multi-Tenant Extensibility (`user_id`)**
   - **Rationale**: The application currently has no user identity or authentication system. Conversation ownership is explicitly deferred. However, repository method signatures include an optional `user_id: Optional[str] = None` parameter (`create_conversation(user_id)`, `get_conversation(conversation_id, user_id)`) so user ownership can later be enforced without breaking interface contracts.

6. **Postponement of Semantic Memory & Vector Embeddings**
   - **Rationale**: Raw conversation history stores *episodic stream data*, not extracted *semantic knowledge*. Semantic memory extraction, vector embeddings, and retrieval-augmented generation (RAG) are intentionally postponed to future sprints.

## Consequences

### Positive
- Strict separation of API, service, repository, and LLM abstraction layers.
- `ChatService` can be tested in memory using fake/mock repository implementations without database setup.
- Updated conversation timestamps accurately reflect last activity.
- Future-proof signature design for user ownership and multi-tenancy.

### Limitations & Future Work
- **Authentication / Multi-Tenancy**: Current endpoints operate without user ownership boundaries. User authentication and multi-tenant authorization will be enforced prior to production deployment.
- **Context Window Management**: The current implementation loads full conversation history. Summarization, sliding window truncation, and token budgeting will be added as conversation lengths grow.
