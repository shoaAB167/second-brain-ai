# ADR 004: Persistent Conversation Memory Architecture

## Status
Accepted

## Context
Second Brain AI requires a mechanism to store and retrieve multi-turn conversation context across user interactions. Without persistent memory, each request to the chat API would be isolated, losing all context from prior exchanges. 

As we lay the foundation for a future personal intelligence assistant, we need a reliable, structured persistence model for conversation history before introducing advanced cognitive capabilities such as semantic memory, vector search, or autonomous agents.

## Decision

We decide to implement relational conversation persistence in PostgreSQL using SQLAlchemy 2.x (asyncpg) and Alembic migrations.

### Key Architectural Choices

1. **Relational Database (PostgreSQL)**
   - **Rationale**: PostgreSQL provides ACID compliance, strong data integrity, efficient indexing, and native support for UUIDs and JSON/array primitives. It is the industry standard for reliable relational storage and can seamlessly co-locate vector extensions (e.g. `pgvector`) in future phases.

2. **Entity Separation: `Conversation` and `Message`**
   - **Rationale**: Decoupling the conversation thread from individual messages enforces a 1-to-many relationship (`Conversation` 1 ──< `Message`). This allows independent message querying, pagination, indexing, soft-deletions, and metadata tracking per conversation without mutating monolithic JSON documents.

3. **Role Enforcement (`user`, `assistant`, `system`)**
   - **Rationale**: Messages use a constrained `MessageRole` Enum to enforce provider-independent chat message roles strictly at both the application and database schema layers.

4. **Chronological Message Ordering**
   - **Rationale**: Queries retrieve messages ordered by `created_at ASC` backed by a compound index `(conversation_id, created_at)`. This guarantees chronological context ordering when feeding history back to the LLM abstraction.

5. **Postponement of Semantic Memory & Vector Embeddings**
   - **Rationale**: Raw conversation history stores *episodic stream data*, not extracted *semantic knowledge*. Semantic memory extraction, vector embeddings, and retrieval-augmented generation (RAG) are intentionally postponed to future sprints. Storing clean relational messages now establishes the raw data source required for future offline indexing and background extraction pipelines.

## Consequences

### Positive
- Strict separation of API, service, repository, and LLM abstraction layers.
- Fast, indexed message retrieval for active chat threads.
- Clean database migrations using Alembic without runtime `create_all()` side effects.
- Future-proof foundation for semantic memory extraction and vector search extensions (`pgvector`).

### Limitations & Future Work
- **Authentication / Multi-Tenancy**: Current endpoints operate without user ownership boundaries. User authentication and multi-tenant authorization will be enforced prior to production deployment.
- **Context Window Management**: The current implementation loads full conversation history. Summarization, sliding window truncation, and token budgeting will be added as conversation lengths grow.
