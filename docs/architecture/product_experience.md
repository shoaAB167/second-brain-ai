# Personal AI Product Experience Architecture

## 1. Overview & Product Philosophy

The Personal AI Product Experience provides a calm, intelligent, personal, trustworthy, and interactive interface for Second Brain AI.

Second Brain AI is designed to feel like a dedicated personal workspace and conversational companion grounded in your long-term memory.

### Core Architecture

```
                    ┌── TEXT ────────┐
                    │                │
USER ───────────────┼── VOICE ───────┼──→ PERSONAL BRAIN (Memory + Context + Agent + Tools)
                    │                │
                    └── JARVIS MODE ─┘
```

All interfaces interact with the identical backend intelligence pipeline:
1. **User Authentication & Isolation**: User session scoped via JWT Bearer tokens.
2. **Context Retrieval**: Dimension-aware personal context and experiences.
3. **Agent Orchestration**: `PersonalAgent` with deterministic tools and memory search.
4. **Streaming Response**: Server-Sent Events (SSE) yielded incrementally to the client.

---

## 2. Voice & JARVIS Interactive System

### ⚡ JARVIS Live Voice Mode
- **Animated Neural Orb**: Multi-layered concentric rotating rings, pulsing energy core, and dynamic soundwave frequency visualizers.
- **Visual State Reactions**:
  - 🔵 **Listening**: Electric cyan/blue pulse with reactive audio waveforms.
  - 🟣 **Thinking / Processing**: Deep violet/indigo orbital rotation.
  - 🟢/🟠 **Speaking**: Resonating acoustic pulse with real-time live captions.
  - ⚪ **Idle**: Calm breathing aura.
- **Continuous Hands-Free Conversational Loop (Turn-Taking)**:
  - User speaks → auto-silence detection submits prompt after ~1.5s pause.
  - Second Brain streams response and speaks it aloud with natural TTS.
  - When speech concludes (`onSpeechEnd`), microphone automatically re-opens for hands-free dialogue.
  - Instant interruptibility: clicking the orb or tapping `Interrupt` immediately halts TTS speech and begins listening.

### Voice Input (Speech-to-Text)
- Uses the native browser Web Speech API (`SpeechRecognition` / `webkitSpeechRecognition`).
- Explicit state machine: `IDLE` → `LISTENING` → `PROCESSING` → `IDLE`.
- Displays live transcription in the composer, allowing the user to edit or cancel before sending.
- Gracefully degrades to standard text input when unsupported or when permissions are denied.

### Voice Output (Text-to-Speech)
- Uses the browser `window.speechSynthesis` API.
- Cleans and strips markdown formatting (code blocks, asterisks, URLs) to produce natural spoken speech.
- Provides per-message speaker toggle (`Listen` / `Stop`) with active speaking indicators.
- Automatically cancels active speech synthesis when a new streaming generation starts or when the user navigates.

---

## 3. Safe Personal Context Visibility

Second Brain's primary differentiator is that it remembers the user. The interface surfaces this grounding subtly and respectfully:

- **Subtle Context Indicator**: Assistant responses grounded in memory display a subtle badge (e.g. `✨ Using 3 memories`).
- **Expandable Grounding**: Users can inspect the high-level topics (e.g., `Work`, `Health`, `Study`) that guided the response.
- **Privacy & Safety Invariant**: Internal implementation details—such as raw vector similarity scores, embedding dimensions, database UUIDs, ranking weights, and raw memory retrieval metadata—are **never** exposed to the user or rendered into client-side executable contexts.

---

## 4. Streaming & Conversation State

- **SSE Streaming**: Subscribes to `POST /api/v1/chat/stream` using an asynchronous stream reader.
- **Incremental Rendering**: Assistant message tokens render progressively with a blinking cursor.
- **Interruption & Control**: Users can click `Stop` at any point during streaming to abort the active `AbortController`.
- **Thread Persistence**: Conversation thread IDs are managed in local session storage and preserved across page interactions.
- **Retry Mechanism**: If a network or backend failure occurs, the user can click `Retry` to seamlessly re-dispatch the prompt.

---

## 5. Responsive Design & Accessibility

- **Breakpoints**: Engineered for desktop (1440px), laptop (1024px), tablet (768px), and mobile (390px).
- **Mobile Experience**: Compact composer, touch-friendly touch targets, hidden non-critical badges, and vertical layout.
- **Accessibility**: Semantic HTML elements (`<main>`, `<header>`, `<button>`, `role="article"`), keyboard navigation (`Enter` to send, `Shift+Enter` for newlines), visible focus states, and ARIA labels.
