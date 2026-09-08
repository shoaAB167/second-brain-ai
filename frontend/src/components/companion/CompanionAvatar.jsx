import React from "react";

/**
 * CompanionAvatar Component.
 * An ethereal, luminous fantasy-style female companion presence.
 * Features soft celestial breathing rings, starlight shimmering aura,
 * and harmonious energy waves reflecting conversational presence.
 */
export function CompanionAvatar({ state = "idle", isSpeaking = false, isListening = false, isThinking = false, onClick }) {
  let activeState = "idle";
  if (isSpeaking) {
    activeState = "speaking";
  } else if (isThinking) {
    activeState = "thinking";
  } else if (isListening) {
    activeState = "listening";
  }

  return (
    <div
      className={`companion-avatar-wrap ${activeState}`}
      onClick={onClick}
      role="button"
      tabIndex={0}
      aria-label={`Companion Avatar: ${activeState}`}
      title={`Companion presence: ${activeState}`}
    >
      {/* Outer Ethereal Radiant Glow */}
      <div className="avatar-ambient-glow" />

      {/* Celestial Harmonic Rings */}
      <div className="avatar-ring avatar-ring-outer" />
      <div className="avatar-ring avatar-ring-mid" />
      <div className="avatar-ring avatar-ring-inner" />

      {/* Harmonic Starlight Waves */}
      <div className="avatar-waves-container" aria-hidden="true">
        <span className="starlight-bar bar-1" />
        <span className="starlight-bar bar-2" />
        <span className="starlight-bar bar-3" />
        <span className="starlight-bar bar-4" />
        <span className="starlight-bar bar-5" />
      </div>

      {/* Central Luminous Core Presence */}
      <div className="avatar-luminous-core">
        <div className="core-shimmer" />
        <span className="avatar-core-glyph" aria-hidden="true">
          {activeState === "listening" && "✨"}
          {activeState === "thinking" && "💫"}
          {activeState === "speaking" && "🌸"}
          {activeState === "idle" && "✧"}
        </span>
      </div>
    </div>
  );
}
