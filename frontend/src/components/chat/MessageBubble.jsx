import React from "react";
import { ContextBadge } from "../memory/ContextBadge";
import { renderMarkdown } from "../../utils/markdownRenderer";
import { MessageRole } from "../../types/chat";

export function MessageBubble({
  message,
  isStreamingLastMessage,
  isSpeaking,
  onToggleSpeak,
  voiceOutputSupported,
}) {
  const isUser = message.role === MessageRole.USER;
  const isAssistant = message.role === MessageRole.ASSISTANT;

  return (
    <div
      className={`message-row ${isUser ? "user-row" : "assistant-row"}`}
      role="article"
      aria-label={`${isUser ? "You" : "Second Brain"} message`}
    >
      <div className="message-avatar" aria-hidden="true">
        {isUser ? "👤" : "🧠"}
      </div>

      <div className={`message-bubble ${isUser ? "bubble-user" : "bubble-assistant"}`}>
        <div className="message-header-row">
          <span className="message-sender">
            {isUser ? "You" : "Second Brain"}
          </span>
          {message.timestamp && (
            <span className="message-timestamp">{message.timestamp}</span>
          )}
        </div>

        {/* Subtle Personal Context Badge */}
        {isAssistant && message.personalContext && (
          <ContextBadge personalContext={message.personalContext} />
        )}

        <div className="message-text">
          {message.content ? (
            isAssistant ? (
              renderMarkdown(message.content)
            ) : (
              <span className="user-message-content">{message.content}</span>
            )
          ) : isStreamingLastMessage ? (
            <span className="typing-indicator" aria-label="Thinking...">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </span>
          ) : (
            <span className="empty-message-text">No response generated</span>
          )}

          {isStreamingLastMessage && message.content && (
            <span className="blinking-cursor" aria-hidden="true">▍</span>
          )}
        </div>

        {/* Voice Output Speaker Action */}
        {isAssistant && !isStreamingLastMessage && message.content && voiceOutputSupported && (
          <div className="message-actions">
            <button
              type="button"
              className={`btn-action-speaker ${isSpeaking ? "speaking" : ""}`}
              onClick={() => onToggleSpeak && onToggleSpeak(message.content, message.id)}
              aria-label={isSpeaking ? "Stop reading response" : "Read response aloud"}
              title={isSpeaking ? "Stop reading" : "Read aloud"}
            >
              {isSpeaking ? (
                <>
                  <span className="speaker-icon">🔊</span>
                  <span className="speaker-label">Stop</span>
                </>
              ) : (
                <>
                  <span className="speaker-icon">🔈</span>
                  <span className="speaker-label">Listen</span>
                </>
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
