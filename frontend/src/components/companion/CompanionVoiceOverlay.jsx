import React from "react";
import { CompanionAvatar } from "./CompanionAvatar";
import { renderMarkdown } from "../../utils/markdownRenderer";
import { DEFAULT_COMPANION_IDENTITY } from "../../types/companion";

export function CompanionVoiceOverlay({
  isOpen,
  onClose,
  companionIdentity = DEFAULT_COMPANION_IDENTITY,
  isListening,
  isSpeaking,
  isThinking,
  transcript,
  interimTranscript,
  lastAssistantMessage,
  handsFreeEnabled,
  onToggleHandsFree,
  onStartListening,
  onStopListening,
  onInterrupt,
}) {
  if (!isOpen) return null;

  const currentSpokenText = (transcript + " " + interimTranscript).trim();
  const contextMeta = lastAssistantMessage?.personalContext;
  const companionName = companionIdentity.name || "Aria";

  return (
    <div className="companion-overlay-backdrop" role="dialog" aria-modal="true" aria-label={`${companionName} Voice Sanctuary`}>
      <div className="companion-sanctuary-card">
        {/* Top Header */}
        <div className="sanctuary-header">
          <div className="companion-identity-badge">
            <span className="companion-presence-dot" />
            <div className="companion-header-text">
              <span className="companion-name-title">{companionName}</span>
              <span className="companion-role-sub">Personal Companion</span>
            </div>
          </div>

          <div className="sanctuary-top-actions">
            {contextMeta && contextMeta.memory_count > 0 && (
              <span className="sanctuary-context-badge" title="Grounded in long-term memory">
                ✨ Guided by {contextMeta.memory_count} {contextMeta.memory_count === 1 ? "memory" : "memories"}
              </span>
            )}
            <button
              type="button"
              className="sanctuary-close-btn"
              onClick={onClose}
              aria-label={`Close voice sanctuary and return to chat`}
              title="Return to text conversation"
            >
              ✕ Return to Chat
            </button>
          </div>
        </div>

        {/* Central Avatar Stage */}
        <div className="companion-center-stage">
          <CompanionAvatar
            isSpeaking={isSpeaking}
            isListening={isListening}
            isThinking={isThinking}
            onClick={isSpeaking ? onInterrupt : isListening ? onStopListening : onStartListening}
          />

          <div className="companion-presence-status">
            {isSpeaking && <span className="presence-label speaking">{companionName} is speaking...</span>}
            {isThinking && !isSpeaking && <span className="presence-label thinking">{companionName} is thinking...</span>}
            {isListening && <span className="presence-label listening">Listening to you...</span>}
            {!isSpeaking && !isThinking && !isListening && (
              <span className="presence-label idle">Tap avatar to speak</span>
            )}
          </div>
        </div>

        {/* Live Spoken Captions */}
        <div className="companion-captions-box">
          {/* User Spoken Prompt */}
          {currentSpokenText && (
            <div className="companion-caption-item user-caption-item">
              <span className="caption-tag">You</span>
              <p className="caption-body">{currentSpokenText}</p>
            </div>
          )}

          {/* Companion Response Caption */}
          {lastAssistantMessage && lastAssistantMessage.content && (
            <div className="companion-caption-item companion-caption-item">
              <span className="caption-tag companion-tag">{companionName}</span>
              <div className="caption-body markdown-caption-body">
                {renderMarkdown(lastAssistantMessage.content)}
              </div>
            </div>
          )}
        </div>

        {/* Bottom Sanctuary Controls */}
        <div className="sanctuary-controls-bar">
          {/* Hands-Free Toggle */}
          <button
            type="button"
            className={`btn-sanctuary-ctrl ${handsFreeEnabled ? "btn-handsfree-active" : ""}`}
            onClick={onToggleHandsFree}
            title={handsFreeEnabled ? "Hands-Free Auto Dialogue Active" : "Click to enable Hands-Free Auto Dialogue"}
          >
            <span className="btn-icon-indicator">{handsFreeEnabled ? "🌸" : "⚪"}</span>
            <span className="btn-text-content">
              Hands-Free: {handsFreeEnabled ? "Continuous" : "Manual"}
            </span>
          </button>

          {/* Push to Talk / Stop Listening */}
          <button
            type="button"
            className={`btn-sanctuary-ctrl ${isListening ? "btn-listening-state" : "btn-primary-voice"}`}
            onClick={isListening ? onStopListening : onStartListening}
            disabled={isThinking || isSpeaking}
          >
            <span className="btn-icon-indicator">{isListening ? "🛑" : "🎙️"}</span>
            <span className="btn-text-content">
              {isListening ? "Done Speaking" : "Talk with " + companionName}
            </span>
          </button>

          {/* Interrupt Button */}
          {isSpeaking && (
            <button
              type="button"
              className="btn-sanctuary-ctrl btn-interrupt-voice"
              onClick={onInterrupt}
              title="Interrupt speech and start speaking"
            >
              <span className="btn-icon-indicator">⏹</span>
              <span className="btn-text-content">Interrupt</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
