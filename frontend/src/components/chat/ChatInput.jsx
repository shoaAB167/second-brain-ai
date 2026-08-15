import React, { useState } from "react";

export const ChatInput = ({ onSendMessage, isStreaming }) => {
  const [text, setText] = useState("");

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    const trimmed = text.trim();
    if (trimmed && !isStreaming) {
      onSendMessage(trimmed);
      setText("");
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <form className="chat-input-form" onSubmit={handleSubmit}>
      <div className="input-container">
        <textarea
          className="chat-textarea"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            isStreaming
              ? "Second Brain AI is generating a response..."
              : "Ask Second Brain AI... (Press Enter to send, Shift+Enter for new line)"
          }
          disabled={isStreaming}
          rows={1}
          aria-label="Chat prompt input"
        />
        <button
          type="submit"
          className="btn btn-primary send-button"
          disabled={isStreaming || !text.trim()}
          aria-label="Send message"
        >
          {isStreaming ? (
            <span className="spinner" aria-hidden="true" />
          ) : (
            "Send ➔"
          )}
        </button>
      </div>
    </form>
  );
};
