import React from "react";
import { useAuth } from "../../context/AuthContext";

export function Header({
  conversationId,
  onNewChat,
  isStreaming,
  voiceModeEnabled,
  onToggleVoiceMode,
  voiceSupported,
  onOpenJarvis,
}) {
  const { isAuthenticated, userEmail, logout, openAuthModal } = useAuth();

  return (
    <header className="chat-header">
      <div className="header-brand">
        <div className="brand-title-group">
          <h1 className="header-title">Second Brain</h1>
          <span className="brand-tagline">Personal Intelligence</span>
        </div>
        {conversationId && (
          <span className="conversation-badge" title={`Thread ID: ${conversationId}`}>
            Thread: {conversationId.slice(0, 8)}
          </span>
        )}
      </div>

      <div className="header-actions">
        {/* JARVIS Interactive Voice Mode Trigger */}
        {voiceSupported && (
          <button
            type="button"
            className="btn-jarvis-launch"
            onClick={onOpenJarvis}
            title="Launch JARVIS Live Voice Experience"
            aria-label="Launch JARVIS Voice System"
          >
            <span className="jarvis-pulse-dot" />
            <span className="jarvis-label">⚡ JARVIS</span>
          </button>
        )}

        {/* Optional Voice Conversation Mode Toggle */}
        {voiceSupported && (
          <button
            type="button"
            className={`btn-voice-mode ${voiceModeEnabled ? "active" : ""}`}
            onClick={onToggleVoiceMode}
            aria-pressed={voiceModeEnabled}
            title={
              voiceModeEnabled
                ? "Voice Mode Active: Assistant will read responses aloud"
                : "Enable Auto Voice Mode"
            }
          >
            <span className="voice-mode-icon" aria-hidden="true">
              {voiceModeEnabled ? "🔊" : "🔈"}
            </span>
            <span className="voice-mode-label">Voice Mode</span>
          </button>
        )}

        <button
          type="button"
          className="btn btn-secondary btn-new-chat"
          onClick={onNewChat}
          disabled={isStreaming}
          aria-label="Start new conversation"
          title="Start a fresh conversation thread"
        >
          + New Chat
        </button>

        {isAuthenticated ? (
          <div className="user-profile-badge">
            <span className="user-avatar-initial" aria-hidden="true">
              {userEmail ? userEmail[0].toUpperCase() : "U"}
            </span>
            <span className="user-email-text" title={userEmail}>
              {userEmail}
            </span>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={logout}
              title="Sign Out"
              aria-label="Sign out of account"
            >
              Sign Out
            </button>
          </div>
        ) : (
          <button
            type="button"
            className="btn btn-primary"
            onClick={openAuthModal}
            aria-label="Sign in or register an account"
          >
            Sign In / Register
          </button>
        )}
      </div>
    </header>
  );
}
