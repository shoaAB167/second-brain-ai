# ADR 005: Real-Time Server-Sent Events (SSE) Streaming Chat Architecture

## Status
Accepted

## Context
Second Brain AI requires low-latency real-time response generation for user prompts. While synchronous chat completion (`POST /api/v1/chat`) works well for short responses, longer completions produce perceived latency while waiting for the full payload to return.

To improve user experience without sacrificing structural conversation persistence, we need a streaming mechanism that delivers tokens to the client as they are generated while maintaining exact conversation history in PostgreSQL.

## Decision

We decide to implement streaming using **Server-Sent Events (SSE)** via HTTP (`POST /api/v1/chat/stream`) with provider-independent abstractions.

### Key Architectural Choices

1. **Server-Sent Events (SSE) over WebSockets**
   - **Rationale**: SSE is built natively on top of HTTP (`text/event-stream`), lightweight, unidirectional (server-to-client), firewall-friendly, and simple to consume in browsers via `EventSource` or `fetch()` body readers.
   - **Future Extensibility**: WebSockets remain a candidate if full bidirectional real-time communication (e.g. real-time audio streams or interactive agent control) is required in future sprints.

2. **End-to-End `stream=True` Pipeline**
   - **LLM Provider**: Invoked with `stream=True` via `litellm.acompletion`, causing the model to yield tokens as they are generated rather than pre-generating the full response.
   - **Backend API**: FastAPI's `StreamingResponse` flushes yielded SSE token events (`data: {"type":"token","content":"..."}\n\n`) instantly over chunked HTTP.
   - **Frontend Browser**: `fetch()` reads the stream using `response.body.getReader()` with `TextDecoder`, rendering tokens incrementally into UI state.

3. **Provider-Independent Streaming Abstraction (`LLMClient.stream_response`)**
   - **Rationale**: `LLMClient` defines `stream_response(messages) -> AsyncGenerator[LLMStreamChunk, None]`.
   - `LLMStreamChunk` encapsulates content deltas, finish reasons, and usage metrics. Provider-specific stream objects (such as LiteLLM chunk iterators) are isolated strictly within `LiteLLMClient`.

4. **Accumulated Assistant Message Persistence Strategy**
   - **Rationale**: Chunks are streamed in real time to the client via SSE (`data: {"type":"token","content":"..."}\n\n`).
   - Token chunks are **not** persisted as individual database rows to avoid database write amplification.
   - The full response is accumulated in memory during the stream and persisted as **exactly one** assistant message to PostgreSQL upon successful completion.

5. **Failure & Transaction Semantics**
   - **User Message**: Persisted to database history *before* invoking the LLM stream.
   - **Assistant Message**: Persisted *only after* the stream completes successfully.
   - **Stream Failures**: If provider errors occur mid-stream, an SSE error payload `data: {"type":"error","message":"..."}\n\n` is yielded. The user prompt remains preserved in database history as an attempted message, but no partial assistant message is created.

6. **Resource Safety & Memory Tradeoffs**
   - Tokens are forwarded to the HTTP response stream immediately without buffering the full response before outputting chunks.
   - Holding the accumulated string in memory for database persistence is lightweight for standard completions. If extremely large text payloads are introduced in future sprints, a chunked temporary persistence buffer strategy can be adopted.

## Consequences

### Positive
- Perceived latency is minimized: first tokens render within milliseconds.
- Strict architecture boundaries are preserved: route handlers and service layers do not import LiteLLM or vendor SDKs.
- Clean separation between synchronous (`/chat`) and streaming (`/chat/stream`) endpoints.
- Database records remain clean with exactly one assistant message per interaction.

### Limitations & Future Work
- **Client Disconnections**: Client-side network drops will abort the response generator loop. Unfinished responses are discarded and not persisted.
- **WebSockets**: If bidirectional multi-modal input (e.g. real-time voice streaming) is needed, a WebSocket gateway can be added alongside SSE.
