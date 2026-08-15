import React from "react";

export const MessageBubble = ({ message, isStreamingLastMessage }) => {
  const isUser = message.role === "user";

  return (
    <div
      className={`message-row ${isUser ? "user-row" : "assistant-row"}`}
      role="region"
      aria-label={`${isUser ? "User" : "Assistant"} message`}
    >
      <div className="message-avatar">
        {isUser ? "👤" : "🧠"}
      </div>
      <div className={`message-bubble ${isUser ? "bubble-user" : "bubble-assistant"}`}>
        <div className="message-sender">
          {isUser ? "You" : "Second Brain AI"}
        </div>
        <div className="message-text">
          {message.content ? (
            message.content
          ) : isStreamingLastMessage ? (
            <span className="typing-indicator" aria-label="Thinking...">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </span>
          ) : (
            <span className="empty-message-text">Empty response</span>
          )}
          {isStreamingLastMessage && message.content && (
            <span className="blinking-cursor" aria-hidden="true">▍</span>
          )}
        </div>
      </div>
    </div>
  );
};
