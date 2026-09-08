import React, { useState } from "react";
import { AuthProvider } from "./context/AuthContext";
import { AuthModal } from "./components/auth/AuthModal";
import { ChatInput } from "./components/chat/ChatInput";
import { Header } from "./components/chat/Header";
import { MessageList } from "./components/chat/MessageList";
import { CompanionVoiceOverlay } from "./components/companion/CompanionVoiceOverlay";
import { useChat } from "./hooks/useChat";
import { useCompanionVoiceSession } from "./hooks/useCompanionVoiceSession";
import { DEFAULT_COMPANION_IDENTITY } from "./types/companion";

function CompanionApp() {
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

  const [companionIdentity] = useState(DEFAULT_COMPANION_IDENTITY);
  const [autoVoiceReadAloud, setAutoVoiceReadAloud] = useState(false);

  const lastAssistantMsg =
    messages.length > 0 && messages[messages.length - 1].role === "assistant"
      ? messages[messages.length - 1]
      : null;

  // Single source of truth voice session state machine
  const companionSession = useCompanionVoiceSession({
    onSendMessage: (text) => sendMessage(text),
    isStreaming,
    lastAssistantMessage: lastAssistantMsg,
    silenceDelayMs: 1500,
  });

  const handleOpenCompanion = () => {
    companionSession.startSession();
  };

  const handleCloseCompanion = () => {
    companionSession.stopSession();
  };

  const handleToggleAutoVoice = () => {
    setAutoVoiceReadAloud((prev) => !prev);
  };

  const handleNewChat = () => {
    companionSession.stopSession();
    startNewChat();
  };

  return (
    <div className="app-container">
      <Header
        conversationId={conversationId}
        onNewChat={handleNewChat}
        isStreaming={isStreaming}
        voiceModeEnabled={autoVoiceReadAloud}
        onToggleVoiceMode={handleToggleAutoVoice}
        voiceSupported={
          companionSession.voiceInput.isSupported &&
          companionSession.voiceOutput.isSupported
        }
        companionName={companionIdentity.name}
        onOpenCompanion={handleOpenCompanion}
      />

      <main className="main-content">
        <MessageList
          messages={messages}
          isStreaming={isStreaming}
          error={error}
          onRetry={retryLastMessage}
          onPromptClick={(prompt) => sendMessage(prompt)}
          voiceOutput={companionSession.voiceOutput}
        />
        <ChatInput
          onSendMessage={(text) => sendMessage(text)}
          onStopStreaming={stopStreaming}
          isStreaming={isStreaming}
          voiceInput={companionSession.voiceInput}
        />
      </main>

      {/* Fullscreen / Modal Companion Sanctuary */}
      <CompanionVoiceOverlay
        isOpen={companionSession.isSessionActive}
        onClose={handleCloseCompanion}
        companionIdentity={companionIdentity}
        isListening={companionSession.isListening}
        isSpeaking={companionSession.isSpeaking}
        isThinking={companionSession.isThinking}
        transcript={companionSession.voiceInput.transcript}
        interimTranscript={companionSession.voiceInput.interimTranscript}
        lastAssistantMessage={lastAssistantMsg}
        handsFreeEnabled={companionSession.handsFreeEnabled}
        onToggleHandsFree={companionSession.toggleHandsFree}
        onStartListening={companionSession.voiceInput.startListening}
        onStopListening={companionSession.voiceInput.stopListening}
        onInterrupt={companionSession.interrupt}
      />

      <AuthModal />
    </div>
  );
}

export function App() {
  return (
    <AuthProvider>
      <CompanionApp />
    </AuthProvider>
  );
}

export default App;
