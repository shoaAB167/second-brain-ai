import { useCallback, useEffect, useRef, useState } from "react";
import { VoiceInputState } from "../types/chat";

/**
 * Hook providing speech-to-text functionality using the native Web Speech API.
 * Supports auto-silence detection for hands-free conversational loops.
 */
export function useVoiceInput({ onFinalTranscript, onSpeechFinished, silenceDelayMs = 1600 } = {}) {
  const [voiceState, setVoiceState] = useState(VoiceInputState.IDLE);
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [error, setError] = useState(null);
  const [audioLevel, setAudioLevel] = useState(0);

  const recognitionRef = useRef(null);
  const isListeningRef = useRef(false);
  const silenceTimerRef = useRef(null);
  const onFinalTranscriptRef = useRef(onFinalTranscript);
  const onSpeechFinishedRef = useRef(onSpeechFinished);
  const accumulatedTranscriptRef = useRef("");

  useEffect(() => {
    onFinalTranscriptRef.current = onFinalTranscript;
  }, [onFinalTranscript]);

  useEffect(() => {
    onSpeechFinishedRef.current = onSpeechFinished;
  }, [onSpeechFinished]);

  const isSupported =
    typeof window !== "undefined" &&
    Boolean(window.SpeechRecognition || window.webkitSpeechRecognition);

  const clearSilenceTimer = () => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
  };

  // Initialize SpeechRecognition instance
  useEffect(() => {
    if (!isSupported) return;

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = navigator.language || "en-US";

    recognition.onstart = () => {
      isListeningRef.current = true;
      setVoiceState(VoiceInputState.LISTENING);
      setError(null);
      setAudioLevel(1);
    };

    recognition.onresult = (event) => {
      clearSilenceTimer();
      setAudioLevel(Math.floor(Math.random() * 3) + 2); // Dynamic reactive audio level

      let currentInterim = "";
      let currentFinal = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const item = event.results[i];
        const text = item[0].transcript;
        if (item.isFinal) {
          currentFinal += text;
        } else {
          currentInterim += text;
        }
      }

      if (currentFinal) {
        accumulatedTranscriptRef.current = accumulatedTranscriptRef.current
          ? `${accumulatedTranscriptRef.current} ${currentFinal}`.trim()
          : currentFinal.trim();

        const latestText = accumulatedTranscriptRef.current;
        setTranscript(latestText);

        if (onFinalTranscriptRef.current) {
          onFinalTranscriptRef.current(latestText);
        }
      }

      setInterimTranscript(currentInterim);

      // Start silence timer for hands-free auto submission
      if (accumulatedTranscriptRef.current || currentInterim) {
        silenceTimerRef.current = setTimeout(() => {
          const totalText = (accumulatedTranscriptRef.current + " " + currentInterim).trim();
          if (totalText && onSpeechFinishedRef.current && isListeningRef.current) {
            onSpeechFinishedRef.current(totalText);
          }
        }, silenceDelayMs);
      }
    };

    recognition.onerror = (event) => {
      clearSilenceTimer();
      isListeningRef.current = false;
      setVoiceState(VoiceInputState.ERROR);
      setAudioLevel(0);

      let userFriendlyError = "An error occurred with voice recognition.";
      switch (event.error) {
        case "not-allowed":
        case "service-not-allowed":
          userFriendlyError = "Microphone access is required for voice input.";
          break;
        case "audio-capture":
          userFriendlyError = "No microphone was found on your device.";
          break;
        case "no-speech":
          userFriendlyError = "No speech was detected. Listening...";
          // Don't treat no-speech as fatal error in continuous loop
          setVoiceState(VoiceInputState.LISTENING);
          return;
        case "network":
          userFriendlyError = "Network error during speech recognition.";
          break;
        case "aborted":
          setVoiceState(VoiceInputState.IDLE);
          return;
        default:
          userFriendlyError = `Voice error: ${event.error}`;
      }

      setError(userFriendlyError);
    };

    recognition.onend = () => {
      clearSilenceTimer();
      isListeningRef.current = false;
      setInterimTranscript("");
      setAudioLevel(0);
      setVoiceState((prev) =>
        prev === VoiceInputState.ERROR ? VoiceInputState.ERROR : VoiceInputState.IDLE
      );
    };

    recognitionRef.current = recognition;

    return () => {
      clearSilenceTimer();
      if (recognitionRef.current && isListeningRef.current) {
        try {
          recognitionRef.current.abort();
        } catch {
          // Ignore cleanup errors
        }
      }
    };
  }, [isSupported, silenceDelayMs]);

  const startListening = useCallback(() => {
    if (!isSupported) {
      setError("Voice input isn't supported in this browser. You can still use text.");
      setVoiceState(VoiceInputState.ERROR);
      return;
    }

    clearSilenceTimer();
    setError(null);
    setTranscript("");
    setInterimTranscript("");
    accumulatedTranscriptRef.current = "";

    try {
      if (recognitionRef.current && !isListeningRef.current) {
        recognitionRef.current.start();
      }
    } catch (err) {
      // If already started or restarting
      try {
        recognitionRef.current.stop();
        setTimeout(() => {
          if (recognitionRef.current) recognitionRef.current.start();
        }, 150);
      } catch {
        setError("Microphone is currently unavailable.");
        setVoiceState(VoiceInputState.ERROR);
      }
    }
  }, [isSupported]);

  const stopListening = useCallback(() => {
    clearSilenceTimer();
    if (recognitionRef.current && isListeningRef.current) {
      try {
        setVoiceState(VoiceInputState.PROCESSING);
        recognitionRef.current.stop();
      } catch {
        // Ignore stop error
      }
    }
  }, []);

  const toggleListening = useCallback(() => {
    if (voiceState === VoiceInputState.LISTENING) {
      stopListening();
    } else {
      startListening();
    }
  }, [voiceState, startListening, stopListening]);

  const resetTranscript = useCallback(() => {
    clearSilenceTimer();
    setTranscript("");
    setInterimTranscript("");
    accumulatedTranscriptRef.current = "";
    setError(null);
  }, []);

  return {
    isSupported,
    voiceState,
    isListening: voiceState === VoiceInputState.LISTENING,
    transcript,
    interimTranscript,
    audioLevel,
    error,
    startListening,
    stopListening,
    toggleListening,
    resetTranscript,
  };
}
