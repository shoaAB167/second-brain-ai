import { useCallback, useEffect, useRef, useState } from "react";
import { useVoiceInput } from "./useVoiceInput";
import { useVoiceOutput } from "./useVoiceOutput";

export const CompanionVoiceState = {
  IDLE: "idle",
  LISTENING: "listening",
  PROCESSING: "processing",
  THINKING: "thinking",
  SPEAKING: "speaking",
  ERROR: "error",
};

/**
 * Unified, deterministic Voice Session State Machine for the Personal AI Companion.
 * Acts as the single source of truth for conversational turn-taking, speech synthesis,
 * microphone lifecycle, and interruption without race conditions or duplicated speech.
 */
export function useCompanionVoiceSession({
  onSendMessage,
  isStreaming,
  lastAssistantMessage,
  silenceDelayMs = 1500,
}) {
  const [sessionState, setSessionState] = useState(CompanionVoiceState.IDLE);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [handsFreeEnabled, setHandsFreeEnabled] = useState(true);

  const sessionStateRef = useRef(sessionState);
  const isSessionActiveRef = useRef(isSessionActive);
  const handsFreeRef = useRef(handsFreeEnabled);
  const lastSpokenMsgIdRef = useRef(null);
  const isStreamingRef = useRef(isStreaming);

  useEffect(() => {
    sessionStateRef.current = sessionState;
  }, [sessionState]);

  useEffect(() => {
    isSessionActiveRef.current = isSessionActive;
  }, [isSessionActive]);

  useEffect(() => {
    handsFreeRef.current = handsFreeEnabled;
  }, [handsFreeEnabled]);

  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  // Hook for Text-to-Speech Output
  const voiceOutput = useVoiceOutput({
    onSpeechStart: () => {
      setSessionState(CompanionVoiceState.SPEAKING);
    },
    onSpeechEnd: () => {
      // If session is active and hands-free is enabled, transition to LISTENING
      if (isSessionActiveRef.current && handsFreeRef.current) {
        setSessionState(CompanionVoiceState.LISTENING);
        setTimeout(() => {
          if (isSessionActiveRef.current && !voiceOutput.isSpeaking) {
            voiceInput.startListening();
          }
        }, 200);
      } else {
        setSessionState(CompanionVoiceState.IDLE);
      }
    },
  });

  // Hook for Speech-to-Text Input
  const voiceInput = useVoiceInput({
    onSpeechFinished: (spokenText) => {
      const trimmed = (spokenText || "").trim();
      if (!trimmed || isStreamingRef.current) return;

      // Transition to PROCESSING then THINKING
      setSessionState(CompanionVoiceState.PROCESSING);
      voiceInput.stopListening();

      if (onSendMessage) {
        setSessionState(CompanionVoiceState.THINKING);
        onSendMessage(trimmed);
      }
    },
    silenceDelayMs,
  });

  // Synchronize thinking state when streaming changes
  useEffect(() => {
    if (isStreaming) {
      voiceOutput.stop();
      setSessionState(CompanionVoiceState.THINKING);
    }
  }, [isStreaming, voiceOutput]);

  // When assistant response completes, trigger speech synthesis (exactly once per message)
  useEffect(() => {
    if (
      !isStreaming &&
      lastAssistantMessage &&
      lastAssistantMessage.role === "assistant" &&
      lastAssistantMessage.content
    ) {
      const msgId = lastAssistantMessage.id;
      if (lastSpokenMsgIdRef.current !== msgId) {
        lastSpokenMsgIdRef.current = msgId;

        if (isSessionActiveRef.current) {
          setSessionState(CompanionVoiceState.SPEAKING);
          voiceOutput.speak(lastAssistantMessage.content, msgId);
        }
      }
    }
  }, [isStreaming, lastAssistantMessage, voiceOutput]);

  const startSession = useCallback(() => {
    setIsSessionActive(true);
    isSessionActiveRef.current = true;
    voiceOutput.stop();
    setSessionState(CompanionVoiceState.LISTENING);

    setTimeout(() => {
      voiceInput.startListening();
    }, 150);
  }, [voiceInput, voiceOutput]);

  const stopSession = useCallback(() => {
    setIsSessionActive(false);
    isSessionActiveRef.current = false;
    voiceOutput.stop();
    voiceInput.stopListening();
    setSessionState(CompanionVoiceState.IDLE);
  }, [voiceInput, voiceOutput]);

  const interrupt = useCallback(() => {
    // Stop ongoing speech or streaming immediately and listen
    voiceOutput.stop();
    setSessionState(CompanionVoiceState.LISTENING);
    setTimeout(() => {
      if (isSessionActiveRef.current) {
        voiceInput.startListening();
      }
    }, 100);
  }, [voiceInput, voiceOutput]);

  const toggleHandsFree = useCallback(() => {
    setHandsFreeEnabled((prev) => !prev);
  }, []);

  return {
    sessionState,
    isSessionActive,
    handsFreeEnabled,
    voiceInput,
    voiceOutput,
    isListening: sessionState === CompanionVoiceState.LISTENING || voiceInput.isListening,
    isSpeaking: sessionState === CompanionVoiceState.SPEAKING || voiceOutput.isSpeaking,
    isThinking: sessionState === CompanionVoiceState.THINKING,
    isProcessing: sessionState === CompanionVoiceState.PROCESSING,
    startSession,
    stopSession,
    interrupt,
    toggleHandsFree,
  };
}
