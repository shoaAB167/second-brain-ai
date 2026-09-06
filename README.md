# Second Brain AI

An intelligent, modular, and personal AI Assistant built with FastAPI, PostgreSQL (pgvector), LiteLLM, and React (Vite).

Second Brain AI is designed to feel like a calm, trustworthy personal workspace that remembers your experiences, adapts to your personal patterns, and reflects on your goals over time.

---

## 🏗️ Unified Brain Architecture

Voice and Text are equal input/output channels accessing the same Personal Brain backend without duplicating agent logic or retrieval pipelines:

```
                    ┌── TEXT ──┐
                    │          │
USER ───────────────┤          ├──→ PERSONAL BRAIN (Memory + Context + Agent + Tools)
                    │          │
                    └── VOICE ─┘
```

Both interfaces share identical:
* **Long-Term Memory & Experiences**: High-dimensional semantic embeddings (pgvector) and emotion-grounded memory evolution.
* **Personal Context Retrieval**: Dimension-aware retrieval layer filtered through canonical `MemoryQualityService`.
* **Personal Agent & Deterministic Tools**: Tool calling, memory search, and structured context grounding.
* **User Isolation & Safety**: JWT Bearer authentication scoping all queries, memories, patterns, and reflections to the authenticated user.

---

## 🚀 Running the Application

### 1. Backend (FastAPI + PostgreSQL)

```bash
# Install dependencies (using uv)
uv sync

# Run database migrations
uv run alembic upgrade head

# Start FastAPI development server
uv run uvicorn personal_ai.main:app --reload --app-dir src
```

* **Backend API Base URL**: `http://127.0.0.1:8000`
* **Interactive API Documentation (Swagger)**: `http://127.0.0.1:8000/docs`

---

### 2. Frontend (React + Vite)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

* **Frontend Web App**: `http://localhost:5173`
* **Environment Configuration**: Configured via `VITE_API_BASE_URL` in `frontend/.env` (defaults to `http://localhost:8000`).

---

## 🎙️ Voice & Accessibility Capabilities

Second Brain utilizes browser-native Web Speech APIs to provide voice interaction without introducing heavyweight audio servers:

* **Voice Input (Speech-to-Text)**:
  * Uses `window.SpeechRecognition` / `window.webkitSpeechRecognition`.
  * Real-time transcription into the message composer.
  * Explicit states: `idle`, `listening`, `processing`, and `error`.
  * Gracefully degrades to standard text input on unsupported browsers.
* **Voice Output (Text-to-Speech)**:
  * Uses `window.speechSynthesis` with markdown stripping for natural narration.
  * User-controlled speaker action on assistant messages (`Listen` / `Stop`).
  * Automatic cancellation of speech synthesis upon new stream generation or navigation.
  * Optional **Voice Mode** toggle to automatically narrate assistant responses.

### Browser Compatibility:
* **Google Chrome / Chromium**: Full speech recognition and synthesis support.
* **Safari / WebKit**: Speech recognition (`webkitSpeechRecognition`) and synthesis supported with microphone permission.
* **Firefox / Other**: Speech synthesis supported; speech recognition falls back gracefully to standard text input.

---

## 🛡️ Privacy & Safe Memory Grounding

Second Brain reinforces trust by surfacing context subtly:
* **Grounding Badge**: Responses grounded in past memory display a subtle indicator (e.g. `✨ Using 3 memories`).
* **Safe Inspection**: Users can inspect the high-level domain topics that informed the answer.
* **Privacy Guardrails**: Raw database UUIDs, internal vector similarity scores, ranking weights, and embedding dimensions are strictly kept internal and never rendered to client-side views.

---

## 📁 Repository Structure

```text
second-brain-ai/
│
├── src/
│   └── personal_ai/
│       ├── agents/       # Personal Agent orchestration & tools
│       ├── api/          # REST & SSE endpoints, CORS, routers
│       ├── application/  # Memory, Patterns, Reflections, People services
│       ├── config/       # Settings & environment config
│       ├── db/           # SQLAlchemy models & pgvector repositories
│       ├── domain/       # Core domain entities & business rules
│       ├── llm/          # Provider-independent LLM gateway
│       └── main.py       # FastAPI application entrypoint
│
├── frontend/             # React + Vite web UI
│   ├── src/
│   │   ├── components/   # ChatInput, Header, MessageBubble, MessageList, ContextBadge, AuthModal
│   │   ├── context/      # AuthContext session provider
│   │   ├── hooks/        # useChat, useVoiceInput, useVoiceOutput, useAuth
│   │   ├── services/     # chatApi (SSE stream parser), authApi
│   │   ├── types/        # TypeScript/JS models and constants
│   │   ├── App.jsx
│   │   ├── index.css     # Design tokens & responsive styles
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── tests/
│   ├── unit/             # Unit test suite
│   └── integration/      # End-to-end integration test suite
│
├── docs/                 # Architecture Decision Records & Design Docs
├── migrations/           # Alembic database migrations
└── README.md
```