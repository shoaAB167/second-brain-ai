# Second Brain AI — Personal Companion Experience

An intelligent, modular, and personal AI Companion built with FastAPI, PostgreSQL (pgvector), LiteLLM, and React (Vite).

Second Brain AI is designed to feel like a warm, calm, intelligent personal companion that remembers your experiences, adapts to your personal patterns, and reflects on your goals over time.

---

## 🌸 Personal Companion Architecture

The Personal AI Companion sits gracefully above the existing Second Brain intelligence backend without duplicating memory or agent logic:

```
                         PERSONAL COMPANION
                                │
              ┌─────────────────┼─────────────────┐
              ↓                 ↓                 ↓
      Companion Identity      Voice Session     Visual Presence
     (Aria / Female / "Sir") (Deterministic SM) (Ethereal Avatar)
              │                 │                 │
              └─────────────────┼─────────────────┘
                                ↓
                        EXISTING AI BRAIN
                                │
                ┌───────────────┼───────────────┐
                ↓               ↓               ↓
             Memory          Context          Tools
```

### Core Companion Capabilities:
* **Configurable Identity**: Default persona **Aria**, featuring female presentation, respectful address as **"Sir"**, and warm, curious, follow-up inquiry style.
* **Deterministic Voice Session State Machine**: Predictable lifecycle (`IDLE` ➔ `LISTENING` ➔ `PROCESSING` ➔ `THINKING` ➔ `SPEAKING` ➔ `LISTENING`) with zero duplicate speech.
* **Ethereal Visual Presence**: Lightweight, fantasy-style celestial avatar responding in real-time to conversational states.
* **Session Resilience & Token Refresh**: Built-in `POST /api/v1/auth/refresh` and automatic 401 retry to ensure voice sessions are never interrupted by normal access token expiry.
* **Grounding & Context Visibility**: Non-intrusive memory badges showing high-level topic grounding without leaking internal vectors or database IDs.

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

## 🎙️ Voice & Browser Compatibility

* **Voice Input (Speech-to-Text)**: Uses `window.SpeechRecognition` / `window.webkitSpeechRecognition` with auto-silence detection and graceful text fallback.
* **Voice Output (Text-to-Speech)**: Uses `window.speechSynthesis` with natural female voice priority and markdown pre-processing.
* **Supported Browsers**: Google Chrome, Microsoft Edge, Safari (with microphone permissions), and Firefox (speech output with text input fallback).

---

## 📁 Repository Structure

```text
second-brain-ai/
│
├── src/
│   └── personal_ai/
│       ├── agents/       # Personal Agent orchestration & tools
│       ├── api/          # REST & SSE endpoints, CORS, routers (auth/refresh, chat/stream)
│       ├── application/  # Memory, Patterns, Reflections, People services
│       ├── config/       # Settings & environment config
│       ├── db/           # SQLAlchemy models & pgvector repositories
│       ├── domain/       # Core domain entities & business rules
│       ├── llm/          # Provider-independent LLM gateway
│       └── main.py       # FastAPI application entrypoint
│
├── frontend/             # React + Vite web UI
│   ├── src/
│   │   ├── components/
│   │   │   ├── chat/     # ChatInput, Header, MessageBubble, MessageList
│   │   │   ├── companion/# CompanionAvatar, CompanionVoiceOverlay
│   │   │   ├── memory/   # ContextBadge
│   │   │   └── auth/     # AuthModal
│   │   ├── context/      # AuthContext session provider
│   │   ├── hooks/        # useChat, useCompanionVoiceSession, useVoiceInput, useVoiceOutput, useAuth
│   │   ├── services/     # chatApi (SSE parser with 401 retry), authApi (with token refresh)
│   │   ├── types/        # companion.js, chat.js
│   │   ├── App.jsx
│   │   ├── index.css     # Calm fantasy-companion aesthetic tokens
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