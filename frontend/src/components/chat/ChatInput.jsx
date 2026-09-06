import React, { useEffect, useRef, useState } from "react";
import { VoiceInputState } from "../../types/chat";

export function ChatInput({
  onSendMessage,
  onStopStreaming,
  isStreaming,
  voiceInput,
}) {
  const [text, setText] = useState("");
  const textareaRef = useRef(null);

  const {
    isSupported: voiceSupported,
    voiceState,
    isListening,
    transcript,
    interimTranscript,
    error: voiceError,
    toggleListening,
    resetTranscript,
  } = voiceInput || {};

  // Sync speech recognition transcript into composer text input
  useEffect(() => {
    if (transcript) {
      setText(transcript);
    }
  }, [transcript]);

  // Autosize textarea height based on content
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      const nextHeight = Math.min(textareaRef.current.scrollHeight, 180);
      textareaRef.current.style.height = `${nextHeight}px`;
    }
  }, [text, interimTranscript]);

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    if (isListening && toggleListening) {
      toggleListening();
    }
    const combinedText = (text || "").trim();
    if (combinedText && !isStreaming) {
      onSendMessage(combinedText);
      setText("");
      if (resetTranscript) {
        resetTranscript();
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="chat-input-wrapper">
      {/* Voice Listening Active Pill Banner */}
      {isListening && (
        <div className="voice-active-banner" role="status" aria-live="polite">
          <span className="voice-pulse-dot" />
          <span className="voice-status-text">
            Listening... {interimTranscript ? `"${interimTranscript}"` : "Speak now"}
          </span>
          <button
            type="button"
            className="btn-voice-done"
            onClick={handleSubmit}
            aria-label="Send spoken message"
          >
            Send Spoken Prompt ➔
          </button>
        </div>
      )}

      {/* Voice Error Notification */}
      {voiceError && (
        <div className="voice-error-toast" role="alert">
          <span className="voice-error-icon">⚠️</span>
          <span className="voice-error-msg">{voiceError}</span>
        </div>
      )}

      <form className="chat-input-form" onSubmit={handleSubmit}>
        <div className={`input-container ${isListening ? "input-listening" : ""}`}>
          {/* Microphone Voice Button */}
          <button
            type="button"
            className={`btn-mic ${isListening ? "mic-active" : ""} ${
              !voiceSupported ? "mic-unsupported" : ""
            }`}
            onClick={toggleListening}
            disabled={isStreaming}
            title={
              !voiceSupported
                ? "Voice input not supported in this browser"
                : isListening
                ? "Click to stop listening"
                : "Click to speak"
            }
            aria-label={
              !voiceSupported
                ? "Voice input unsupported"
                : isListening
                ? "Stop recording voice"
                : "Start voice input"
            }
          >
            <span className="mic-icon" aria-hidden="true">
              {isListening ? "🛑" : "🎤"}
            </span>
          </button>

          <textarea
            ref={textareaRef}
            className="chat-textarea"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              isStreaming
                ? "Second Brain is thinking..."
                : isListening
                ? "Listening to your voice..."
                : "Ask anything... (Enter to send, Shift+Enter for newline)"
            }
            disabled={isStreaming}
            rows={1}
            aria-label="Personal AI message input"
          />

          {/* Send or Stop button */}
          {isStreaming ? (
            <button
              type="button"
              className="btn btn-secondary stop-button"
              onClick={onStopStreaming}
              aria-label="Stop generating response"
              title="Stop generating"
            >
              <span className="stop-icon" aria-hidden="true">⏹</span>
              <span className="btn-label">Stop</span>
            </button>
          ) : (
            <button
              type="submit"
              className="btn btn-primary send-button"
              disabled={!text.trim()}
              aria-label="Send message"
              title="Send message"
            >
              <span className="send-arrow">➔</span>
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
