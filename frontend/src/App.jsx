import React from "react";
import { AuthProvider } from "./context/AuthContext";
import { AuthModal } from "./components/auth/AuthModal";
import { ChatInput } from "./components/chat/ChatInput";
import { Header } from "./components/chat/Header";
import { MessageList } from "./components/chat/MessageList";
import { useChat } from "./hooks/useChat";

function ChatApp() {
  const {
    messages,
    conversationId,
    isStreaming,
    error,
    sendMessage,
    startNewChat,
  } = useChat();

  return (
    <div className="app-container">
      <Header
        conversationId={conversationId}
        onNewChat={startNewChat}
        isStreaming={isStreaming}
      />

      <main className="main-content">
        <MessageList
          messages={messages}
          isStreaming={isStreaming}
          error={error}
        />
        <ChatInput
          onSendMessage={sendMessage}
          isStreaming={isStreaming}
        />
      </main>

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
