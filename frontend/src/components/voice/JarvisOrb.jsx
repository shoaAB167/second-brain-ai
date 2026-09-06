import React from "react";

/**
 * JARVIS Neural Orb Component.
 * Features multi-layered animated concentric rings, a pulsing energy core,
 * and dynamic soundwave frequency visualizers that react to voice states.
 */
export function JarvisOrb({ state = "idle", isSpeaking = false, isListening = false, isStreaming = false, onClick }) {
  // Determine effective visual mode
  let orbMode = "idle";
  if (isSpeaking) {
    orbMode = "speaking";
  } else if (isStreaming) {
    orbMode = "processing";
  } else if (isListening) {
    orbMode = "listening";
  }

  return (
    <div className={`jarvis-orb-container ${orbMode}`} onClick={onClick} role="button" tabIndex={0} aria-label={`JARVIS Orb: ${orbMode}`}>
      {/* Outer energy aura */}
      <div className="orb-aura-glow" />

      {/* Orbital rotating rings */}
      <div className="orb-ring orb-ring-outer" />
      <div className="orb-ring orb-ring-middle" />
      <div className="orb-ring orb-ring-inner" />

      {/* Acoustic waveform frequency bars */}
      <div className="orb-waveform-container" aria-hidden="true">
        <span className="wave-bar wave-1" />
        <span className="wave-bar wave-2" />
        <span className="wave-bar wave-3" />
        <span className="wave-bar wave-4" />
        <span className="wave-bar wave-5" />
        <span className="wave-bar wave-6" />
        <span className="wave-bar wave-7" />
      </div>

      {/* Central neural core */}
      <div className="orb-core">
        <div className="orb-core-pulse" />
        <span className="orb-center-icon">
          {orbMode === "listening" && "🎙️"}
          {orbMode === "processing" && "⚡"}
          {orbMode === "speaking" && "🔊"}
          {orbMode === "idle" && "🧠"}
        </span>
      </div>
    </div>
  );
}
