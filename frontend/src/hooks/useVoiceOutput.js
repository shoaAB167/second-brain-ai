import { useCallback, useEffect, useRef, useState } from "react";
import { VoiceOutputState } from "../types/chat";

/**
 * Utility to strip markdown characters from text for natural speech synthesis.
 */
function cleanTextForSpeech(text) {
  if (!text) return "";
  return text
    .replace(/```[\s\S]*?```/g, "code block omitted.") // Replace multiline code
    .replace(/`([^`]+)`/g, "$1") // Inline code
    .replace(/\*\*([^*]+)\*\*/g, "$1") // Bold
    .replace(/\*([^*]+)\*/g, "$1") // Italic
    .replace(/#+\s+(.*)/g, "$1") // Headers
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1") // Links
    .replace(/^\s*[-*+]\s+/gm, "") // List bullets
    .replace(/>\s*(.*)/g, "$1") // Blockquotes
    .trim();
}

/**
 * Hook providing speech synthesis (text-to-speech) capabilities via window.speechSynthesis.
 * Manages active speaking message, user controls (play/pause/stop), cancellation, and speech lifecycle callbacks.
 */
export function useVoiceOutput({ onSpeechEnd, onSpeechStart } = {}) {
  const [voiceOutputState, setVoiceOutputState] = useState(VoiceOutputState.IDLE);
  const [activeMessageId, setActiveMessageId] = useState(null);

  const isSupported =
    typeof window !== "undefined" && Boolean(window.speechSynthesis);

  const utteranceRef = useRef(null);
  const onSpeechEndRef = useRef(onSpeechEnd);
  const onSpeechStartRef = useRef(onSpeechStart);

  useEffect(() => {
    onSpeechEndRef.current = onSpeechEnd;
  }, [onSpeechEnd]);

  useEffect(() => {
    onSpeechStartRef.current = onSpeechStart;
  }, [onSpeechStart]);

  const stop = useCallback(() => {
    if (!isSupported) return;
    try {
      window.speechSynthesis.cancel();
    } catch {
      // Ignore cancellation error
    }
    utteranceRef.current = null;
    setActiveMessageId(null);
    setVoiceOutputState(VoiceOutputState.IDLE);
  }, [isSupported]);

  const pause = useCallback(() => {
    if (!isSupported) return;
    try {
      window.speechSynthesis.pause();
      setVoiceOutputState(VoiceOutputState.PAUSED);
    } catch {
      // Ignore pause error
    }
  }, [isSupported]);

  const resume = useCallback(() => {
    if (!isSupported) return;
    try {
      window.speechSynthesis.resume();
      setVoiceOutputState(VoiceOutputState.SPEAKING);
    } catch {
      // Ignore resume error
    }
  }, [isSupported]);

  const speak = useCallback(
    (text, messageId = null) => {
      if (!isSupported) return;

      // Stop any ongoing speech first
      stop();

      const cleanedText = cleanTextForSpeech(text);
      if (!cleanedText) {
        if (onSpeechEndRef.current) {
          onSpeechEndRef.current();
        }
        return;
      }

      const utterance = new SpeechSynthesisUtterance(cleanedText);
      utterance.rate = 1.05; // slightly faster, crisp voice cadence
      utterance.pitch = 1.0;

      // Select a natural, crisp voice if available
      const voices = window.speechSynthesis.getVoices();
      const preferredVoice = voices.find(
        (v) =>
          v.lang.startsWith("en") &&
          (v.name.includes("Natural") ||
            v.name.includes("Google") ||
            v.name.includes("Samantha") ||
            v.name.includes("Daniel") ||
            v.name.includes("Arthur") ||
            v.name.includes("Oliver"))
      ) || voices.find((v) => v.lang.startsWith("en"));

      if (preferredVoice) {
        utterance.voice = preferredVoice;
      }

      utterance.onstart = () => {
        setVoiceOutputState(VoiceOutputState.SPEAKING);
        setActiveMessageId(messageId);
        if (onSpeechStartRef.current) {
          onSpeechStartRef.current(messageId);
        }
      };

      utterance.onend = () => {
        setVoiceOutputState(VoiceOutputState.IDLE);
        setActiveMessageId(null);
        utteranceRef.current = null;
        if (onSpeechEndRef.current) {
          onSpeechEndRef.current(messageId);
        }
      };

      utterance.onerror = (event) => {
        setVoiceOutputState(VoiceOutputState.IDLE);
        setActiveMessageId(null);
        utteranceRef.current = null;
        if (event.error !== "interrupted" && event.error !== "canceled") {
          console.warn("Speech synthesis error:", event);
        }
      };

      utteranceRef.current = utterance;
      window.speechSynthesis.speak(utterance);
    },
    [isSupported, stop]
  );

  const toggleSpeak = useCallback(
    (text, messageId) => {
      if (activeMessageId === messageId && voiceOutputState === VoiceOutputState.SPEAKING) {
        stop();
      } else {
        speak(text, messageId);
      }
    },
    [activeMessageId, voiceOutputState, speak, stop]
  );

  // Cancel any ongoing speech when component unmounts
  useEffect(() => {
    return () => {
      if (isSupported) {
        try {
          window.speechSynthesis.cancel();
        } catch {
          // Ignore
        }
      }
    };
  }, [isSupported]);

  return {
    isSupported,
    isSpeaking: voiceOutputState === VoiceOutputState.SPEAKING,
    isPaused: voiceOutputState === VoiceOutputState.PAUSED,
    activeMessageId,
    speak,
    stop,
    pause,
    resume,
    toggleSpeak,
  };
}
