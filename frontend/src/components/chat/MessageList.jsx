import React, { useEffect, useRef } from "react";
import { MessageBubble } from "./MessageBubble";

export const MessageList = ({ messages, isStreaming, error }) => {
  const containerRef = useRef(null);
  const userScrolledUpRef = useRef(false);

  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
    userScrolledUpRef.current = distanceFromBottom > 100;
  };

  useEffect(() => {
    if (!userScrolledUpRef.current && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
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
          <div className="empty-icon">🧠</div>
          <h2>Welcome to Second Brain AI</h2>
          <p>Start a conversation by typing your prompt below.</p>
        </div>
      ) : (
        messages.map((msg, index) => {
          const isLastAssistant =
            isStreaming &&
            index === messages.length - 1 &&
            msg.role === "assistant";

          return (
            <MessageBubble
              key={msg.id}
              message={msg}
              isStreamingLastMessage={isLastAssistant}
            />
          );
        })
      )}

      {error && (
        <div className="error-banner" role="alert">
          <span className="error-icon">⚠️</span>
          <span className="error-message">{error}</span>
        </div>
      )}
    </div>
  );
};
