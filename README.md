# Second Brain AI

An intelligent, modular, and extensible Personal AI Assistant built with FastAPI, PostgreSQL, LiteLLM, and React (TypeScript + Vite).

---

## 🚀 Running the Application

### 1. Backend (FastAPI + PostgreSQL)

```bash
# Install dependencies (using uv)
uv sync

# Run database migrations
uv run alembic upgrade head

# Start development server
uv run uvicorn personal_ai.main:app --reload --app-dir src
```

* **Backend API Base URL**: `http://127.0.0.1:8000`
* **Swagger API Docs**: `http://127.0.0.1:8000/docs`

---

### 2. Frontend (React + TypeScript + Vite)

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
```

* **Frontend Web App**: `http://localhost:5173`
* **API Base URL Configuration**: Configured via `VITE_API_BASE_URL` in `frontend/.env` (defaults to `http://127.0.0.1:8000`).

---

## 🏗️ System Architecture

```
                    Browser
                       │
                       ▼
                 React Chat UI (frontend/)
                       │
                       ▼
                   chatApi (services/chatApi.ts)
                       │
                 HTTP + SSE (fetch + getReader)
                       │
                       ▼
                FastAPI Backend (src/personal_ai/)
                       │
                   ChatService (services/chat_service.py)
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
 ConversationRepository       LLMClient (ABC)
            │                     │
            ▼                     ▼
       PostgreSQL            LiteLLMClient
            │                     │
      (asyncpg)                   ▼
                             LLM Provider (Gemini, OpenAI, etc.)
```

---

## 📁 Repository Structure

```text
second-brain-ai/
│
├── src/
│   └── personal_ai/
│       ├── api/          # REST & SSE endpoints, CORS, routers
│       ├── config/       # Pydantic settings & env config
│       ├── core/         # Telemetry, logging, exceptions
│       ├── db/           # ORM models, session setup, repositories
│       ├── llm/          # Provider-independent LLM gateway & LiteLLM
│       ├── models/       # Pydantic DTOs & domain models
│       ├── services/     # Business logic orchestration
│       └── main.py       # FastAPI app creation & middleware
│
├── frontend/             # React + TypeScript + Vite web UI
│   ├── src/
│   │   ├── components/   # ChatWindow, MessageList, MessageBubble, ChatInput, Header
│   │   ├── hooks/        # useChat custom state hook
│   │   ├── services/     # chatApi fetch-based SSE parser
│   │   ├── types/        # TypeScript interfaces for Chat
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
│
├── tests/
│   ├── unit/             # Backend & LLM unit tests
│   └── integration/      # End-to-end API & DB tests
│
├── docs/                 # ADR design documents (ADR 001 - ADR 005)
├── migrations/           # Alembic database migrations
├── pyproject.toml        # Backend dependencies & configuration
├── uv.lock
└── README.md
```

---

## ✨ Current Capabilities

- **Real-Time Token Streaming**: `POST /api/v1/chat/stream` streams completions via Server-Sent Events (SSE).
- **Persistent Conversation Memory**: PostgreSQL tracks conversations and messages across sessions.
- **Provider-Independent LLM Layer**: Supports OpenAI, Gemini, Anthropic, Ollama, DeepSeek, and OpenRouter via LiteLLM without code changes.
- **Active Thread Persistence**: The web UI automatically persists the active `conversation_id` in `localStorage` (`second_brain_conversation_id`).
- **Interactive Controls**: Features a "New Chat" button, multiline input, auto-scroll, loading indicators, and error banners.

---

## ⚠️ Current Limitations

- **No Authentication**: User accounts and identity systems are not yet implemented.
- **No Long-Term Personal Memory**: Entity extraction and persistent facts are deferred to future sprints.
- **No Vector Search / RAG**: Embeddings, vector databases (pgvector), and document retrieval are deferred to future sprints.
- **No Tools / Agents**: Function calling, external integrations, and autonomous agents are deferred to future sprints.
- **No UI History Restoration Yet**: The UI restores `conversation_id` on refresh to continue the backend thread, but full history GET fetching will be added in a future PR.