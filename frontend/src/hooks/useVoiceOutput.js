import { useCallback, useEffect, useRef, useState } from "react";
import { VoiceOutputState } from "../types/chat";

/**
 * Utility to strip markdown characters and code blocks from text for natural speech synthesis.
 */
function cleanTextForCompanionSpeech(text) {
  if (!text) return "";
  return text
    .replace(/```[\s\S]*?```/g, "I have shared a code snippet in our notes.") // Replace code blocks
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
 * Hook providing speech synthesis (text-to-speech) capabilities with female persona voice selection.
 * Manages active speaking message, user controls (play/pause/stop), interruption, and voiceschanged lifecycle.
 */
export function useVoiceOutput({ onSpeechEnd, onSpeechStart } = {}) {
  const [voiceOutputState, setVoiceOutputState] = useState(VoiceOutputState.IDLE);
  const [activeMessageId, setActiveMessageId] = useState(null);
  const [availableVoices, setAvailableVoices] = useState([]);

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

  // Load and cache voices, handling asynchronous voiceschanged in Chrome/Edge/Safari
  useEffect(() => {
    if (!isSupported) return;

    const updateVoices = () => {
      const voices = window.speechSynthesis.getVoices();
      if (voices && voices.length > 0) {
        setAvailableVoices(voices);
      }
    };

    updateVoices();

    if (window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = updateVoices;
    }

    return () => {
      if (window.speechSynthesis.onvoiceschanged === updateVoices) {
        window.speechSynthesis.onvoiceschanged = null;
      }
    };
  }, [isSupported]);

  const selectFemaleCompanionVoice = useCallback(() => {
    const voices = availableVoices.length > 0 ? availableVoices : (typeof window !== "undefined" && window.speechSynthesis ? window.speechSynthesis.getVoices() : []);
    if (!voices || voices.length === 0) return null;

    // Preference list for warm, clear female voices
    const preferredNames = [
      "Samantha",
      "Victoria",
      "Karen",
      "Serena",
      "Google UK English Female",
      "Google US English",
      "Microsoft Zira",
      "Fiona",
      "Moira",
      "Natural",
    ];

    for (const name of preferredNames) {
      const match = voices.find(
        (v) => v.lang.startsWith("en") && v.name.toLowerCase().includes(name.toLowerCase())
      );
      if (match) return match;
    }

    // Fallback: any English voice
    return voices.find((v) => v.lang.startsWith("en")) || voices[0] || null;
  }, [availableVoices]);

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

      const cleanedText = cleanTextForCompanionSpeech(text);
      if (!cleanedText) {
        if (onSpeechEndRef.current) {
          onSpeechEndRef.current(messageId);
        }
        return;
      }

      const utterance = new SpeechSynthesisUtterance(cleanedText);
      utterance.rate = 1.0;
      utterance.pitch = 1.05; // Gentle, warm companion pitch

      const femaleVoice = selectFemaleCompanionVoice();
      if (femaleVoice) {
        utterance.voice = femaleVoice;
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
    [isSupported, stop, selectFemaleCompanionVoice]
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
    availableVoices,
    speak,
    stop,
    pause,
    resume,
    toggleSpeak,
  };
}
