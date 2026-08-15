import React from "react";

export const Header = ({ conversationId, onNewChat }) => {
  return (
    <header className="chat-header">
      <div className="header-brand">
        <h1 className="header-title">Second Brain AI</h1>
        {conversationId && (
          <span className="conversation-badge" title={`ID: ${conversationId}`}>
            Active Thread: {conversationId.slice(0, 8)}...
          </span>
        )}
      </div>
      <button
        type="button"
        className="btn btn-secondary"
        onClick={onNewChat}
        aria-label="Start new conversation"
      >
        + New Chat
      </button>
    </header>
  );
};
