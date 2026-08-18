import React from "react";
import { useAuth } from "../../context/AuthContext";

export const Header = ({ conversationId, onNewChat }) => {
  const { isAuthenticated, userEmail, logout, openAuthModal } = useAuth();

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

      <div className="header-actions">
        <button
          type="button"
          className="btn btn-secondary"
          onClick={onNewChat}
          aria-label="Start new conversation"
        >
          + New Chat
        </button>

        {isAuthenticated ? (
          <div className="user-profile-badge">
            <span className="user-email-text" title={userEmail}>
              {userEmail}
            </span>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={logout}
              title="Sign Out"
            >
              Sign Out
            </button>
          </div>
        ) : (
          <button
            type="button"
            className="btn btn-primary"
            onClick={openAuthModal}
          >
            Sign In / Register
          </button>
        )}
      </div>
    </header>
  );
};
