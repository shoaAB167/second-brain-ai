import React from "react";
import { JarvisOrb } from "./JarvisOrb";
import { renderMarkdown } from "../../utils/markdownRenderer";

export function JarvisVoiceOverlay({
  isOpen,
  onClose,
  isListening,
  isSpeaking,
  isStreaming,
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

  return (
    <div className="jarvis-overlay-backdrop" role="dialog" aria-modal="true" aria-label="JARVIS Voice Experience">
      <div className="jarvis-hud-container">
        {/* Top HUD Header */}
        <div className="jarvis-hud-header">
          <div className="hud-brand">
            <span className="hud-indicator-dot" />
            <div className="hud-title-wrap">
              <span className="hud-title">JARVIS SYSTEM</span>
              <span className="hud-sub">SECOND BRAIN NEURAL INTERFACE</span>
            </div>
          </div>

          <div className="hud-status-badges">
            {contextMeta && contextMeta.memory_count > 0 && (
              <span className="hud-memory-badge" title="Active memory context">
                ✨ {contextMeta.memory_count} MEMORIES ACTIVE
              </span>
            )}
            <button
              type="button"
              className="hud-close-btn"
              onClick={onClose}
              aria-label="Exit JARVIS Mode"
              title="Return to standard chat interface"
            >
              ✕ EXIT
            </button>
          </div>
        </div>

        {/* Central Neural Orb Area */}
        <div className="jarvis-center-stage">
          <JarvisOrb
            isSpeaking={isSpeaking}
            isListening={isListening}
            isStreaming={isStreaming}
            onClick={isSpeaking ? onInterrupt : isListening ? onStopListening : onStartListening}
          />

          <div className="jarvis-state-status">
            {isSpeaking && <span className="status-label speaking">JARVIS IS SPEAKING...</span>}
            {isStreaming && !isSpeaking && <span className="status-label thinking">CONSULTING SECOND BRAIN...</span>}
            {isListening && <span className="status-label listening">LISTENING TO YOU...</span>}
            {!isSpeaking && !isStreaming && !isListening && (
              <span className="status-label idle">READY — TAP ORB TO SPEAK</span>
            )}
          </div>
        </div>

        {/* Dynamic Speech & Response Captions */}
        <div className="jarvis-captions-display">
          {/* User Spoken Live Transcript */}
          {currentSpokenText && (
            <div className="caption-bubble user-caption">
              <span className="caption-sender">YOU</span>
              <p className="caption-text">{currentSpokenText}</p>
            </div>
          )}

          {/* JARVIS Live Streaming / Spoken Response */}
          {lastAssistantMessage && lastAssistantMessage.content && (
            <div className="caption-bubble jarvis-caption">
              <span className="caption-sender">JARVIS</span>
              <div className="caption-text markdown-caption">
                {renderMarkdown(lastAssistantMessage.content)}
              </div>
            </div>
          )}
        </div>

        {/* Bottom HUD Controls */}
        <div className="jarvis-hud-controls">
          {/* Hands-Free Toggle */}
          <button
            type="button"
            className={`hud-ctrl-btn ${handsFreeEnabled ? "btn-active" : ""}`}
            onClick={onToggleHandsFree}
            title="Auto-listen for user speech after JARVIS finishes speaking"
          >
            <span className="btn-icon">{handsFreeEnabled ? "🟢" : "⚪"}</span>
            <span className="btn-text">
              Hands-Free: {handsFreeEnabled ? "ACTIVE" : "OFF"}
            </span>
          </button>

          {/* Mic Action */}
          <button
            type="button"
            className={`hud-ctrl-btn ${isListening ? "btn-listening" : "btn-primary-action"}`}
            onClick={isListening ? onStopListening : onStartListening}
            disabled={isStreaming || isSpeaking}
          >
            <span className="btn-icon">{isListening ? "🛑" : "🎙️"}</span>
            <span className="btn-text">
              {isListening ? "Stop Listening" : "Speak Now"}
            </span>
          </button>

          {/* Interrupt Button */}
          {isSpeaking && (
            <button
              type="button"
              className="hud-ctrl-btn btn-interrupt"
              onClick={onInterrupt}
              title="Interrupt speech and start speaking"
            >
              <span className="btn-icon">⏹</span>
              <span className="btn-text">Interrupt</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
