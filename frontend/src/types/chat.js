/**
 * Centralized types, constants, and models for Second Brain AI Chat & Voice Interface.
 */

export const MessageRole = {
  USER: "user",
  ASSISTANT: "assistant",
  SYSTEM: "system",
};

export const StreamEventType = {
  CONTEXT: "context",
  TOKEN: "token",
  DONE: "done",
  ERROR: "error",
};

export const VoiceInputState = {
  IDLE: "idle",
  LISTENING: "listening",
  PROCESSING: "processing",
  ERROR: "error",
};

export const VoiceOutputState = {
  IDLE: "idle",
  SPEAKING: "speaking",
  PAUSED: "paused",
};

/**
 * @typedef {Object} PersonalContextMeta
 * @property {number} [memory_count] - Count of memories utilized
 * @property {string[]} [topics] - High-level domain/topic summaries
 */

/**
 * @typedef {Object} ChatMessage
 * @property {string} id - Unique identifier
 * @property {'user' | 'assistant' | 'system'} role - Role of the sender
 * @property {string} content - Message text or markdown
 * @property {string} [timestamp] - ISO timestamp or formatted time
 * @property {PersonalContextMeta} [personalContext] - Attached memory context summary
 * @property {boolean} [isError] - Whether this message represents a failure
 * @property {boolean} [isStreaming] - Whether the message is actively receiving chunks
 */
