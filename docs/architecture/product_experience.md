# Personal AI Companion Experience Architecture

## 1. Overview & Companion Philosophy

The Personal AI Companion Experience provides a calm, warm, intelligent, and human-like interface for Second Brain AI.

Rather than feeling like a generic chatbot or a sci-fi robot/HUD, Second Brain presents a consistent, graceful personal presence—configured by default as **Aria**, a female companion who addresses the user respectfully as **"Sir"**, engages with genuine curiosity, and asks thoughtful follow-up questions.

### Core Architecture

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

The companion is an interface layer that sits **above** the existing intelligence architecture:
1. **User Authentication & Session Resilience**: User session scoped via JWT Bearer tokens with automatic token refresh (`POST /api/v1/auth/refresh`).
2. **Context Retrieval**: Dimension-aware personal context and memory evolution.
3. **Agent Orchestration**: `PersonalAgent` informed by companion persona guidelines without duplicating agent logic.
4. **Streaming Response**: Server-Sent Events (SSE) yielded incrementally to the client.

---

## 2. Companion Identity & Conversational Guidelines

The companion identity is defined in `frontend/src/types/companion.js` and supplied to the existing backend agent pipeline via structured system instructions:

- **Name**: Configurable (default: `"Aria"`).
- **Presentation**: Female personal companion.
- **User Address**: Respectful address as `"Sir"` when natural in conversation.
- **Tone & Style**: Warm, attentive, curious, concise when appropriate, asking meaningful follow-up questions to understand the user's intent.
- **Honesty Invariant**: The companion never claims consciousness or physical embodiment.

---

## 3. Deterministic Voice Session State Machine

Voice interactions are orchestrated by `useCompanionVoiceSession`, which acts as the single source of truth for conversational turn-taking:

```
                  ┌───────────────┐
                  │     IDLE      │
                  └───────┬───────┘
                          │ startSession()
                          ▼
                  ┌───────────────┐
   ┌─────────────►│   LISTENING   │◄──────────────────────┐
   │              └───────┬───────┘                       │
   │                      │ user finishes speech          │
   │                      ▼                               │
   │              ┌───────────────┐                       │
   │              │  PROCESSING   │                       │
   │              └───────┬───────┘                       │
   │                      │ prompt sent                   │
   │                      ▼                               │
   │              ┌───────────────┐                       │
   │              │   THINKING    │                       │
   │              └───────┬───────┘                       │
   │                      │ stream finishes               │
   │                      ▼                               │
   │              ┌───────────────┐                       │
   │ interrupt()  │   SPEAKING    │                       │
   └──────────────┴───────┬───────┘                       │
                          │ speech ends (hands-free ON)   │
                          └───────────────────────────────┘
```

### Key Safeguards:
- **Duplicate Speech Prevention**: Each assistant response message ID is tracked and spoken exactly once.
- **Stale Closure Protection**: Recognition instances and timers use React `useRef` to eliminate closure-capture bugs.
- **Microphone Lockout**: Microphone listening is explicitly blocked while Text-to-Speech is speaking.
- **Instant Interruption**: User speech or clicking `Interrupt` instantly cancels speech synthesis and begins listening.

---

## 4. Visual Presence & Ethereal Avatar

The companion features a lightweight, fantasy-style celestial avatar (`CompanionAvatar.jsx`):
- **IDLE**: Gentle breathing starlight aura with soft luminous pulsing.
- **LISTENING**: Attentive focus pulse with radiant celestial waves.
- **THINKING**: Soft shimmering celestial swirl.
- **SPEAKING**: Harmonic luminous resonance synchronized with voice narration.

---

## 5. Token Refresh & Session Resilience

- **Backend**: `POST /api/v1/auth/refresh` issues a fresh access token for authenticated user sessions.
- **Frontend**: `chatApi.js` intercepts 401 Unauthorized errors and performs an automatic silent token refresh via deduplicated `refreshToken()` before retrying the chat request once.
- Voice sessions are never abruptly severed due to routine token expiration.

---

## 6. Safe Personal Context Visibility

- **Context Grounding**: Assistant responses display a subtle indicator (e.g. `✨ Guided by 3 memories`).
- **Privacy Guarantee**: Raw database UUIDs, internal vector similarity scores, ranking weights, and embedding dimensions are never exposed.
