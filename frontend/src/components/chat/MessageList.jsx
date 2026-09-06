import React, { useEffect, useRef } from "react";
import { MessageBubble } from "./MessageBubble";
import { MessageRole } from "../../types/chat";

const STARTER_PROMPTS = [
  "What should I focus on today?",
  "What have we discussed recently?",
  "Help me reflect on my productivity and routines.",
];

export function MessageList({
  messages,
  isStreaming,
  error,
  onRetry,
  onPromptClick,
  voiceOutput,
}) {
  const containerRef = useRef(null);
  const userScrolledUpRef = useRef(false);

  const { isSpeaking, activeMessageId, toggleSpeak, isSupported: voiceOutputSupported } =
    voiceOutput || {};

  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
    userScrolledUpRef.current = distanceFromBottom > 100;
  };

  useEffect(() => {
    if (!userScrolledUpRef.current && containerRef.current) {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: isStreaming ? "auto" : "smooth",
      });
    }
  }, [messages, isStreaming]);

  return (
    <div
      className="message-list"
      ref={containerRef}
      onScroll={handleScroll}
      aria-live="polite"
      aria-atomic="false"
    >
      {messages.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon-wrapper">
            <span className="empty-brain-icon" aria-hidden="true">🧠</span>
          </div>
          <h2 className="empty-title">Second Brain AI</h2>
          <p className="empty-subtitle">
            Your personal AI grounded in your experiences, patterns, and context.
          </p>

          <div className="starter-prompts-container">
            <span className="starters-label">Suggestions to get started:</span>
            <div className="starter-chips">
              {STARTER_PROMPTS.map((prompt, index) => (
                <button
                  key={index}
                  type="button"
                  className="starter-chip"
                  onClick={() => onPromptClick && onPromptClick(prompt)}
                >
                  <span className="chip-arrow">➔</span> {prompt}
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        messages.map((msg, index) => {
          const isLastAssistant =
            isStreaming &&
            index === messages.length - 1 &&
            msg.role === MessageRole.ASSISTANT;

          const isCurrentlySpeaking =
            isSpeaking && activeMessageId === msg.id;

          return (
            <MessageBubble
              key={msg.id || index}
              message={msg}
              isStreamingLastMessage={isLastAssistant}
              isSpeaking={isCurrentlySpeaking}
              onToggleSpeak={toggleSpeak}
              voiceOutputSupported={voiceOutputSupported}
            />
          );
        })
      )}

      {error && (
        <div className="error-banner" role="alert">
          <span className="error-icon" aria-hidden="true">⚠️</span>
          <div className="error-content">
            <span className="error-message">{error}</span>
            {onRetry && (
              <button
                type="button"
                className="btn-error-retry"
                onClick={onRetry}
                aria-label="Retry sending prompt"
              >
                Retry ↺
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
