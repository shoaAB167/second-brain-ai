import React, { useEffect, useRef, useState } from "react";
import { AuthProvider } from "./context/AuthContext";
import { AuthModal } from "./components/auth/AuthModal";
import { ChatInput } from "./components/chat/ChatInput";
import { Header } from "./components/chat/Header";
import { MessageList } from "./components/chat/MessageList";
import { JarvisVoiceOverlay } from "./components/voice/JarvisVoiceOverlay";
import { useChat } from "./hooks/useChat";
import { useVoiceInput } from "./hooks/useVoiceInput";
import { useVoiceOutput } from "./hooks/useVoiceOutput";

function ChatApp() {
  const {
    messages,
    conversationId,
    isStreaming,
    error,
    sendMessage,
    stopStreaming,
    startNewChat,
    retryLastMessage,
  } = useChat();

  const [voiceModeEnabled, setVoiceModeEnabled] = useState(false);
  const [jarvisOpen, setJarvisOpen] = useState(false);
  const [handsFreeEnabled, setHandsFreeEnabled] = useState(true);

  const jarvisOpenRef = useRef(jarvisOpen);
  const handsFreeRef = useRef(handsFreeEnabled);

  useEffect(() => {
    jarvisOpenRef.current = jarvisOpen;
  }, [jarvisOpen]);

  useEffect(() => {
    handsFreeRef.current = handsFreeEnabled;
  }, [handsFreeEnabled]);

  // Hook for Text-to-Speech
  const voiceOutput = useVoiceOutput({
    onSpeechEnd: () => {
      // If in JARVIS Mode with Hands-Free active, auto-listen for next user turn
      if (jarvisOpenRef.current && handsFreeRef.current) {
        setTimeout(() => {
          voiceInput.startListening();
        }, 300);
      }
    },
  });

  // Hook for Speech-to-Text with hands-free auto submission
  const voiceInput = useVoiceInput({
    onFinalTranscript: (spokenText) => {
      // Live transcript update
    },
    onSpeechFinished: (completedSpeech) => {
      // In JARVIS Mode or Auto Voice Mode, automatically submit user's spoken sentence
      if ((jarvisOpenRef.current || voiceModeEnabled) && completedSpeech.trim() && !isStreaming) {
        voiceInput.stopListening();
        sendMessage(completedSpeech.trim());
      }
    },
    silenceDelayMs: 1500,
  });

  // Cancel any active speech when a new stream starts
  useEffect(() => {
    if (isStreaming) {
      voiceOutput.stop();
    }
  }, [isStreaming, voiceOutput]);

  // When stream finishes in JARVIS or Voice Mode, auto-speak the assistant response
  const lastAssistantMsg = messages.length > 0 ? messages[messages.length - 1] : null;
  const lastAssistantIdRef = useRef(null);

  useEffect(() => {
    if (!isStreaming && lastAssistantMsg && lastAssistantMsg.role === "assistant" && lastAssistantMsg.content) {
      if (lastAssistantIdRef.current !== lastAssistantMsg.id) {
        lastAssistantIdRef.current = lastAssistantMsg.id;
        if (jarvisOpen || voiceModeEnabled) {
          voiceOutput.speak(lastAssistantMsg.content, lastAssistantMsg.id);
        }
      }
    }
  }, [isStreaming, jarvisOpen, voiceModeEnabled, lastAssistantMsg, voiceOutput]);

  const handleOpenJarvis = () => {
    setJarvisOpen(true);
    setHandsFreeEnabled(true);
    // Start listening immediately when entering JARVIS mode
    setTimeout(() => {
      voiceInput.startListening();
    }, 250);
  };

  const handleCloseJarvis = () => {
    setJarvisOpen(false);
    voiceOutput.stop();
    if (voiceInput.isListening) {
      voiceInput.stopListening();
    }
  };

  const handleInterrupt = () => {
    voiceOutput.stop();
    stopStreaming();
    setTimeout(() => {
      voiceInput.startListening();
    }, 150);
  };

  const handleToggleVoiceMode = () => {
    if (voiceModeEnabled) {
      setVoiceModeEnabled(false);
      voiceOutput.stop();
    } else {
      setVoiceModeEnabled(true);
    }
  };

  const handleNewChat = () => {
    voiceOutput.stop();
    if (voiceInput.isListening) {
      voiceInput.stopListening();
    }
    startNewChat();
  };

  return (
    <div className="app-container">
      <Header
        conversationId={conversationId}
        onNewChat={handleNewChat}
        isStreaming={isStreaming}
        voiceModeEnabled={voiceModeEnabled}
        onToggleVoiceMode={handleToggleVoiceMode}
        voiceSupported={voiceInput.isSupported && voiceOutput.isSupported}
        onOpenJarvis={handleOpenJarvis}
      />

      <main className="main-content">
        <MessageList
          messages={messages}
          isStreaming={isStreaming}
          error={error}
          onRetry={retryLastMessage}
          onPromptClick={sendMessage}
          voiceOutput={voiceOutput}
        />
        <ChatInput
          onSendMessage={sendMessage}
          onStopStreaming={stopStreaming}
          isStreaming={isStreaming}
          voiceInput={voiceInput}
        />
      </main>

      {/* Immersive JARVIS Voice Experience */}
      <JarvisVoiceOverlay
        isOpen={jarvisOpen}
        onClose={handleCloseJarvis}
        isListening={voiceInput.isListening}
        isSpeaking={voiceOutput.isSpeaking}
        isStreaming={isStreaming}
        transcript={voiceInput.transcript}
        interimTranscript={voiceInput.interimTranscript}
        lastAssistantMessage={lastAssistantMsg?.role === "assistant" ? lastAssistantMsg : null}
        handsFreeEnabled={handsFreeEnabled}
        onToggleHandsFree={() => setHandsFreeEnabled(!handsFreeEnabled)}
        onStartListening={voiceInput.startListening}
        onStopListening={voiceInput.stopListening}
        onInterrupt={handleInterrupt}
      />

      <AuthModal />
    </div>
  );
}

export function App() {
  return (
    <AuthProvider>
      <ChatApp />
    </AuthProvider>
  );
}

export default App;
